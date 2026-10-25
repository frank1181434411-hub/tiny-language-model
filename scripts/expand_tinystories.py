"""
从固定官方版本抽取完整故事，扩充本地训练集；限制下载量，复用词表和历史验证集。
随机字节块抽样不是对全库故事的均匀抽样；边界故事丢弃，原始块缓存便于复核。
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import time
import urllib.request
from tiny_lm.data import load_token_data
from tiny_lm.preprocess import process_corpus
from tiny_lm.tokenizer.tokenizer import BPETokenizer

REVISION="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
BASE=f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}"
SIZES={"train":1924281556,"valid":19447282}
EOS="<|endoftext|>"


def digest(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


def normalize(text:str)->str:
    return text.replace("\r\n","\n").replace("\r","\n").strip()


def complete_stories(raw:bytes)->list[str]:
    # 先按 ASCII 分隔符截出完整记录，再解码，避免块边界切断 UTF-8 字符。
    return [text for part in raw.split(EOS.encode())[1:-1] if (text:=normalize(part.decode("utf-8")))]


def words(text:str)->tuple[str,...]:
    return tuple(re.findall(r"\w+",text.casefold()))


class OverlapIndex:
    """
    对大小写/标点规范化全文去重，并检查词 5-gram 集合 Jaccard >= 0.8。
    这只是可复现的词面近似重复规则，不代表语义去重。
    """
    def __init__(self):
        self.exact=set()
        self.sizes=[]
        self.postings=defaultdict(list)

    def check(self,text:str,add:bool=False)->bool:
        tokens=words(text)
        key=digest(" ".join(tokens).encode())
        shingles=set(zip(*(tokens[i:] for i in range(5)),strict=False))
        duplicate=key in self.exact
        counts=Counter(index for shingle in shingles for index in self.postings.get(shingle,()))
        if not duplicate:
            duplicate=any(5*overlap>=4*(len(shingles)+self.sizes[index]-overlap) for index,overlap in counts.items())
        if add:
            index=len(self.sizes)
            self.exact.add(key)
            self.sizes.append(len(shingles))
            for shingle in shingles:
                self.postings[shingle].append(index)
        return duplicate


def fetch_candidates(split:str,cache:Path,seed:int,blocks:int,block_bytes:int)->tuple[list[str],list[dict]]:
    rng=random.Random(seed)
    records={}
    provenance=[]
    # 每个等宽区间取一个随机块，避免只取文件开头；全部读取总量有上限。
    width=SIZES[split]//blocks
    if width<block_bytes:
        raise ValueError("range strata must be larger than download blocks")
    for index in range(blocks):
        start=rng.randint(index*width,(index+1)*width-block_bytes)
        end=start+block_bytes-1
        path=cache/f"{split}-{start}-{end}.raw"
        if path.exists():
            raw=path.read_bytes()
        else:
            url=f"{BASE}/TinyStories-{split}.txt?download=true&range={start}-{end}"
            request=urllib.request.Request(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"})
            with urllib.request.urlopen(request,timeout=60) as response:
                if response.status!=206 or response.headers.get("Content-Range")!=f"bytes {start}-{end}/{SIZES[split]}":
                    raise ValueError("server did not honor the bounded byte range; refusing full download")
                raw=response.read(block_bytes+1)
            if len(raw)!=block_bytes:
                raise ValueError("downloaded range length does not match requested range")
            path.write_bytes(raw)
        if len(raw)!=block_bytes:
            raise ValueError("cached range has an invalid size")
        stories=complete_stories(raw)
        for text in stories:
            records.setdefault(digest(text.encode()),text)
        provenance.append({"split":split,"start":start,"end":end,"sha256":digest(raw),"complete_records":len(stories),"cache_file":str(path.resolve())})
        print(f"Downloaded/cached {split} block {index+1}/{blocks}: {len(records)} unique complete stories",flush=True)
    ordered=sorted(records.values(),key=lambda text:digest((str(seed)+"\0"+text).encode()))
    return ordered,provenance


def encode_split(name:str,stories:list[str],output:Path,tokenizer:BPETokenizer)->dict:
    directory=output/(name+"-shards")
    directory.mkdir()
    paths=[]
    for start in range(0,len(stories),500):
        path=directory/f"{start//500:04d}.txt"
        path.write_bytes("".join(text+EOS+"\n" for text in stories[start:start+500]).encode())
        paths.append(path)
    path=process_corpus(paths,output/(name+".bin"),tokenizer)
    metadata=json.loads(path.with_suffix(".bin.json").read_text(encoding="utf-8"))
    data=load_token_data(path,tokenizer=tokenizer)
    try:
        offset=0
        for source in metadata["sources"]:
            end=offset+source["num_tokens"]
            if tokenizer.decode(data[offset:end],errors="strict")!=Path(source["path"]).read_bytes().decode():
                raise ValueError("decoded shard does not match source text")
            offset=end
        eos_count=int((data==tokenizer.special_to_id[EOS]).sum())
        if eos_count!=len(stories):
            raise ValueError("each story must contain exactly one EOS")
        return {"num_stories":len(stories),"num_tokens":len(data),"eos_count":eos_count,"sha256":metadata["sha256"],"roundtrip_verified":True,"story_sha256":[digest(text.encode()) for text in stories]}
    finally:
        data._mmap.close()


def main():
    parser=argparse.ArgumentParser(description="Expand TinyStories with bounded official range downloads; do not train a model.")
    parser.add_argument("--baseline",type=Path,default=Path("data/tinystories-local-v1"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--cache-dir",type=Path,default=Path("downloads/tinystories-ranges-v1"))
    parser.add_argument("--train-stories",type=int,default=50000)
    parser.add_argument("--heldout-stories",type=int,default=1000)
    args=parser.parse_args()
    if args.train_stories<6133 or args.heldout_stories<1:
        raise ValueError("train-stories must be at least 6133 and heldout-stories must be positive")
    started=time.perf_counter()
    baseline=args.baseline.resolve(strict=True)
    original={p.name:digest(p.read_bytes()) for p in baseline.iterdir() if p.is_file()}
    old_train=[normalize(s) for s in (baseline/"train.txt").read_text(encoding="utf-8").split(EOS) if normalize(s)]
    old_val=[normalize(s) for s in (baseline/"val.txt").read_text(encoding="utf-8").split(EOS) if normalize(s)]
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    (args.cache_dir/"README.source.md").write_bytes(urllib.request.urlopen(f"{BASE}/README.md",timeout=30).read())
    candidates,valid_ranges=fetch_candidates("valid",args.cache_dir,43,2,1024*1024)
    old_index=OverlapIndex()
    for text in old_train+old_val:
        old_index.check(text,add=True)
    heldout=[]
    for text in candidates:
        if not old_index.check(text,add=True):
            heldout.append(text)
        if len(heldout)==args.heldout_stories:
            break
    if len(heldout)!=args.heldout_stories:
        raise ValueError("not enough disjoint held-out stories in bounded sample")
    protected=OverlapIndex()
    for text in old_val+heldout:
        protected.check(text,add=True)
    # 先审计历史训练集；若它与保留验证集近似重复，不能悄悄改变对照基线。
    if any(protected.check(text) for text in old_train):
        raise ValueError("baseline train overlaps protected evaluation under the near-duplicate rule; review the split before expansion")
    candidates,train_ranges=fetch_candidates("train",args.cache_dir,42,16,4*1024*1024)
    train=list(old_train)
    seen={" ".join(words(text)) for text in train}
    rejected={"exact_train_duplicates":0,"evaluation_overlap":0}
    for text in candidates:
        key=" ".join(words(text))
        if key in seen:
            rejected["exact_train_duplicates"]+=1
        elif protected.check(text):
            rejected["evaluation_overlap"]+=1
        else:
            seen.add(key)
            train.append(text)
        if len(train)==args.train_stories:
            break
    if len(train)!=args.train_stories:
        raise ValueError("not enough eligible training stories in bounded sample")
    train.sort(key=lambda text:digest(("42\0"+text).encode()))
    for name in ["tokenizer.json","val.txt","val.bin","val.bin.json"]:
        shutil.copyfile(baseline/name,output/name)
    tokenizer=BPETokenizer.from_file(output/"tokenizer.json")
    manifest={"format":"tiny-lm-expanded-tinystories-v1","repository":"roneneldan/TinyStories","revision":REVISION,"license":"cdla-sharing-1.0","sampling":"stratified random byte blocks; discard both edge fragments; seeded SHA256 ordering of complete normalized stories; not uniform over stories","normalization":"CRLF/CR to LF; strip outer whitespace; retain internal paragraphs","overlap_rule":"casefold word-normalized exact match or word 5-gram set Jaccard >= 0.8 against evaluation; normalized exact dedup within train","limitations":"no semantic dedup; no near-dedup within training; partial source hashes only","baseline_files_sha256":original,"range_downloads":valid_ranges+train_ranges,"download_bytes":sum(r["end"]-r["start"]+1 for r in valid_ranges+train_ranges),"rejected":rejected,"retained_baseline_train":len(old_train),"splits":{}}
    print(f"Selection complete: train={len(train)}, historical validation={len(old_val)}, heldout={len(heldout)}. Encoding in 500-story shards...",flush=True)
    for name,stories in [("train",train),("heldout",heldout)]:
        manifest["splits"][name]=encode_split(name,stories,output,tokenizer)
        print(f"Verified {name}: {manifest['splits'][name]['num_tokens']} tokens",flush=True)
    validation=load_token_data(output/"val.bin",tokenizer=tokenizer)
    manifest["splits"]["val"]={"num_stories":len(old_val),"num_tokens":len(validation),"byte_identical_to_baseline":True}
    validation._mmap.close()
    for name,expected in original.items():
        if digest((baseline/name).read_bytes())!=expected:
            raise ValueError("baseline file changed during preparation")
    manifest["elapsed_seconds"]=time.perf_counter()-started
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(output),"seconds":manifest["elapsed_seconds"],"verified":True}),flush=True)


if __name__=="__main__":
    main()
