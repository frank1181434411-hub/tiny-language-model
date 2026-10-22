from collections import Counter
from itertools import product
import builtins
import os
import random
import subprocess
import sys
import pytest
from tiny_lm.tokenizer.bpe import train_bpe
from tiny_lm.tokenizer.rules import iter_pretokens,split_special_tokens
from tiny_lm.tokenizer.serialization import load_tokenizer,save_tokenizer

pytest.importorskip("regex")


def reference_bpe(words,vocab_size,special_tokens):
    # 独立慢速参考：每轮重新统计所有序列，重建合并后的列表。
    sequences=[[bytes([b]) for b in word.encode("utf-8")] for word in words]
    merges=[]
    for _ in range(vocab_size-256-len(special_tokens)):
        counts=Counter(pair for sequence in sequences for pair in zip(sequence,sequence[1:],strict=False))
        if not counts:
            break
        best=max(counts,key=lambda pair:(counts[pair],pair))
        merges.append(best)
        updated=[]
        for sequence in sequences:
            result=[]
            i=0
            while i<len(sequence):
                if tuple(sequence[i:i+2])==best:
                    result.append(b"".join(best))
                    i+=2
                else:
                    result.append(sequence[i])
                    i+=1
            updated.append(result)
        sequences=updated
    tokens=[bytes([b]) for b in range(256)]+[b"".join(pair) for pair in merges]+[token.encode("utf-8") for token in special_tokens]
    return dict(enumerate(tokens)),merges


def write_corpus(tmp_path,text):
    path=tmp_path/"corpus.txt"
    path.write_bytes(text.encode("utf-8"))
    return path


@pytest.mark.parametrize("text,words",[
    ("",[]),
    ("a",["a"]),
    ("aaaaaaa",["aaaaaaa"]),
    ("abababababa",["abababababa"]),
    ("abc1abc",["abc","1","abc"]),
    ("we're here!\n",["we","'re"," here","!","\n"]),
    ("你好 世界🙂\t\r\n",["你好"," 世界","🙂","\t\r\n"]),
    ("a  b\r\nb",["a"," "," b","\r","\n","b"]),
    ("ab<S>cd<S>",["ab","cd"]),
])
def test_matches_reference(tmp_path,text,words):
    assert list(iter_pretokens(text,["<S>"]))==words
    assert train_bpe(write_corpus(tmp_path,text),300,["<S>"])==reference_bpe(words,300,["<S>"])


@pytest.mark.parametrize("seed",range(40))
def test_randomized_incremental_counts(tmp_path,seed):
    rng=random.Random(seed)
    words=["".join(rng.choices("abc",k=rng.randint(1,30))) for _ in range(60)]
    words+=words[:20]
    text="\n".join(words)
    assert train_bpe(write_corpus(tmp_path,text),310,[])==reference_bpe(words,310,[])


def test_all_short_binary_words(tmp_path):
    words=["".join(chars) for size in range(1,9) for chars in product("ab",repeat=size)]
    assert train_bpe(write_corpus(tmp_path,"\n".join(words)),400,[])==reference_bpe(words,400,[])


def test_tie_break_and_id_order(tmp_path):
    vocab,merges=train_bpe(write_corpus(tmp_path,"ab\ncd"),259,["<S>"])
    assert merges==[(b"c",b"d"),(b"a",b"b")]
    assert vocab==dict(enumerate([bytes([i]) for i in range(256)]+[b"cd",b"ab",b"<S>"]))


@pytest.mark.parametrize("specials",[["<X>","<X>tail"],["<X>tail","<X>"]])
def test_longest_special_match(tmp_path,specials):
    text="<X>tail<X>tail"
    assert list(split_special_tokens(text,specials))==[("<X>tail",True),("<X>tail",True)]
    vocab,merges=train_bpe(write_corpus(tmp_path,text),280,specials)
    assert merges==[]
    assert [vocab[256],vocab[257]]==[token.encode() for token in specials]


def test_special_metacharacters_and_boundaries(tmp_path):
    specials=["[.*]","<S>"]
    text="a[.*]b<S><S>c"
    assert list(split_special_tokens(text,specials))==[("a",False),("[.*]",True),("b",False),("<S>",True),("<S>",True),("c",False)]
    assert train_bpe(write_corpus(tmp_path,text),300,specials)[1]==[]


def test_zero_merges_and_early_stop(tmp_path):
    path=write_corpus(tmp_path,"abc")
    assert len(train_bpe(path,256,[])[0])==256
    vocab,merges=train_bpe(path,1000,[])
    assert len(vocab)==258
    assert len(merges)==2


def test_train_save_load(tmp_path):
    specials=["<结束>","a"]
    vocab,merges=train_bpe(write_corpus(tmp_path,"你好你好\r\na<结束>世界🙂🙂"),290,specials)
    path=tmp_path/"tokenizer.json"
    save_tokenizer(path,vocab,merges,specials)
    assert load_tokenizer(path)==(vocab,merges,specials)


@pytest.mark.parametrize("size,error",[(True,TypeError),(256.0,TypeError),("300",TypeError),(None,TypeError),(-1,ValueError),(0,ValueError),(255,ValueError),(256,ValueError)])
def test_invalid_size_before_file_read(tmp_path,size,error):
    with pytest.raises(error,match="vocab_size"):
        train_bpe(tmp_path/"missing.txt",size,["<S>"])


@pytest.mark.parametrize("specials,error",[(None,TypeError),("<S>",TypeError),(("<S>",),TypeError),([1],TypeError),([""],ValueError),(["x","x"],ValueError),(["\ud800"],ValueError)])
def test_invalid_specials_before_file_read(tmp_path,specials,error):
    with pytest.raises(error,match="special_tokens"):
        train_bpe(tmp_path/"missing.txt",300,specials)


def test_missing_file_and_invalid_utf8(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_bpe(tmp_path/"missing.txt",256,[])
    path=tmp_path/"invalid.txt"
    path.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        train_bpe(path,256,[])


def test_missing_optional_dependency(monkeypatch,tmp_path):
    original=builtins.__import__
    def without_regex(name,*args,**kwargs):
        if name=="regex":
            raise ModuleNotFoundError("No module named 'regex'")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",without_regex)
    with pytest.raises(ImportError,match=r"tiny-language-model\[tokenizer\]"):
        train_bpe(write_corpus(tmp_path,"abc"),260,[])


def test_hash_seed_determinism(tmp_path):
    path=write_corpus(tmp_path,"abcabc\naaaaab\nbabababa\n你好 世界🙂")
    code="from tiny_lm.tokenizer.bpe import train_bpe; import sys; print(repr(train_bpe(sys.argv[1],290,[])))"
    outputs=[]
    for seed in ("0","1","17"):
        result=subprocess.run([sys.executable,"-c",code,str(path)],env={**os.environ,"PYTHONHASHSEED":seed},capture_output=True,text=True,check=True)
        outputs.append(result.stdout)
    assert outputs[0]==outputs[1]==outputs[2]
