"""
将已核对的本地 TinyStories 5M 作业样本整理为可复现的小规模训练输入。
仅适用于固定哈希的截断样本：去掉首尾片段，按完整故事划分，BPE 只拟合训练集。
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from tiny_lm.data import load_token_data
from tiny_lm.preprocess import process_corpus
from tiny_lm.tokenizer.bpe import train_bpe
from tiny_lm.tokenizer.tokenizer import BPETokenizer
from tiny_lm.binding import get_tokenizer_fingerprint,token_data_fingerprint

SOURCE_SHA256="09babf3bf1928e32185b133d8b85be949935b576a7c1b861f77107f3b3d82390"
EOS="<|endoftext|>"


def sha256(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


def main():
    parser=argparse.ArgumentParser(description="Prepare the verified local TinyStories 5M fixture without changing the source.")
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    started=time.perf_counter()
    source=args.source.resolve(strict=True)
    raw=source.read_bytes()
    if sha256(raw)!=SOURCE_SHA256:
        raise ValueError("source SHA-256 does not match the verified local 5M fixture; do not apply its edge trimming to another dataset")
    # 显式统一换行并去掉各故事外围空白，保留内部段落。
    parts=raw.decode("utf-8").replace("\r\n","\n").replace("\r","\n").split(EOS)
    stories=[]
    seen=set()
    duplicate_count=0
    for index,part in enumerate(parts[1:-1],start=1):
        text=part.strip()
        if not text:
            continue
        if text in seen:
            duplicate_count+=1
            continue
        seen.add(text)
        stories.append((index,text))
    if len(stories)<20:
        raise ValueError("at least 20 complete unique stories are required")
    # 用 seed 和内容哈希排序，避免依赖平台随机数实现；验证集取向上取整的 5%。
    stories.sort(key=lambda item:(sha256((str(args.seed)+"\0"+item[1]).encode("utf-8")),item[0]))
    validation_count=(len(stories)+19)//20
    splits={"train":stories[validation_count:],"val":stories[:validation_count]}
    if {text for _,text in splits["train"]}&{text for _,text in splits["val"]}:
        raise ValueError("training and validation story contents must be disjoint")
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest={"format":"tiny-lm-local-tinystories-v1","source":{"path":str(source),"sha256":SOURCE_SHA256,"num_bytes":len(raw)},"python_version":platform.python_version(),"seed":args.seed,"split_method":"sort by SHA256(str(seed) + NUL + normalized story); first ceil(N/20) for validation","normalization":"CRLF/CR to LF; strip surrounding story whitespace; preserve internal paragraphs","dropped_edge_fragments":2,"duplicate_stories_removed":duplicate_count,"num_complete_unique_stories":len(stories),"eos_token":EOS,"requested_vocab_size":2048,"splits":{}}
    for name,records in splits.items():
        text="".join(story+EOS+"\n" for _,story in records)
        path=output/(name+".txt")
        path.write_bytes(text.encode("utf-8"))
        manifest["splits"][name]={"num_stories":len(records),"source_segment_indices":[index for index,_ in records],"text_sha256":sha256(text.encode("utf-8")),"text_num_bytes":len(text.encode("utf-8"))}
    print(f"Split complete: train={len(splits['train'])}, validation={len(splits['val'])}. Training BPE on train.txt only...",flush=True)
    bpe_started=time.perf_counter()
    vocab,merges=train_bpe(output/"train.txt",2048,[EOS])
    tokenizer=BPETokenizer(vocab,merges,[EOS])
    tokenizer.save(output/"tokenizer.json")
    tokenizer=BPETokenizer.from_file(output/"tokenizer.json")
    manifest["bpe_seconds"]=time.perf_counter()-bpe_started
    manifest["actual_vocab_size"]=len(tokenizer.vocab)
    manifest["num_merges"]=len(merges)
    manifest["tokenizer_sha256"]=get_tokenizer_fingerprint(tokenizer)
    manifest["tokenizer_file_sha256"]=sha256((output/"tokenizer.json").read_bytes())
    print(f"BPE complete: vocab={len(vocab)}, merges={len(merges)}. Encoding and verifying both splits...",flush=True)
    for name,records in splits.items():
        path=process_corpus(output/(name+".txt"),output/(name+".bin"),tokenizer)
        data=load_token_data(path,tokenizer=tokenizer)
        try:
            text=(output/(name+".txt")).read_bytes().decode("utf-8")
            if tokenizer.decode(data,errors="strict")!=text:
                raise ValueError(f"{name} token decoding does not reproduce the prepared text")
            eos_count=int((data==tokenizer.special_to_id[EOS]).sum())
            if eos_count!=len(records):
                raise ValueError(f"{name} must contain exactly one EOS per story")
            manifest["splits"][name].update({"num_tokens":len(data),"storage_dtype":data.dtype.str,"eos_count":eos_count,"token_fingerprint":token_data_fingerprint(data),"roundtrip_verified":True})
            print(f"{name}: {len(records)} stories, {len(data)} tokens, {eos_count} EOS; hash, ID range and roundtrip verified.",flush=True)
        finally:
            if getattr(data,"_mmap",None) is not None:
                data._mmap.close()
    if sha256(source.read_bytes())!=SOURCE_SHA256:
        raise ValueError("source file changed during preparation")
    manifest["elapsed_seconds"]=time.perf_counter()-started
    # 最后写入完成清单；中途失败的目录保留供排查，重跑须选择新的输出目录。
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output_dir":str(output),"vocab_size":len(vocab),"elapsed_seconds":manifest["elapsed_seconds"],"verified":True}),flush=True)


if __name__=="__main__":
    main()
