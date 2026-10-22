"""
从 Assignment 1 迁移的字节级 BPE 训练。
"""

from pathlib import Path
from collections import defaultdict,Counter
from tiny_lm.tokenizer.rules import iter_pretokens,validate_special_tokens


def train_bpe(input_path:str|Path,vocab_size:int,special_tokens:list[str])->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    """
    读取 UTF-8 文本，返回词表和按优先级排列的合并规则。
    特殊 token 不参与统计，按传入顺序放在词表末尾。
    无可合并字节对时提前结束，实际词表可能小于 vocab_size。
    """
    validate_special_tokens(special_tokens)
    if type(vocab_size) is not int:
        raise TypeError("vocab_size must be an integer")
    if vocab_size<256+len(special_tokens):
        raise ValueError("vocab_size must be at least 256+len(special_tokens)")
    vocab={i:bytes([i]) for i in range(256)}
    cnt_merges=vocab_size-256-len(special_tokens)
    # 保留换行字节；当前仍一次读取整个文本。
    with Path(input_path).open("r",encoding="utf-8",newline="") as f:
        content=f.read()
    pretoken_cnt=Counter()
    for word in iter_pretokens(content,special_tokens):
        pretoken_cnt[tuple(bytes([b]) for b in word.encode("utf-8"))]+=1
    token_seq=[]
    token_freq=[]
    for word,freq in pretoken_cnt.items():
        token_seq.append(list(word))
        token_freq.append(freq)
    pair_freq=defaultdict(int)
    indices=defaultdict(set)
    for i,word in enumerate(token_seq):
        freq=token_freq[i]
        for j in range(len(word)-1):
            pair=(word[j],word[j+1])
            pair_freq[pair]+=freq
            indices[pair].add(i)
    merge_rule=[]
    for _ in range(cnt_merges):
        if not pair_freq:
            break
        # 同频时按字节对的字典序取最大值。
        best_pair=max(pair_freq.items(),key=lambda x:(x[1],x[0]))[0]
        if pair_freq[best_pair]==0:
            break
        merge_rule.append(best_pair)
        new_token=best_pair[0]+best_pair[1]
        updated_indices=list(indices[best_pair])
        for i in updated_indices:
            word=token_seq[i]
            freq=token_freq[i]
            j=0
            while j<len(word)-1:
                if word[j]==best_pair[0] and word[j+1]==best_pair[1]:
                    # 扣除旧邻居，再加入合并后的邻居。
                    if j>0:
                        pre_pair=(word[j-1],word[j])
                        pair_freq[pre_pair]-=freq
                        if pair_freq[pre_pair]==0:
                            del pair_freq[pre_pair]
                    if j<len(word)-2:
                        nex_pair=(word[j+1],word[j+2])
                        pair_freq[nex_pair]-=freq
                        if pair_freq[nex_pair]==0:
                            del pair_freq[nex_pair]
                    word[j]=new_token
                    del word[j+1]
                    if j>0:
                        pre_pair=(word[j-1],word[j])
                        pair_freq[pre_pair]+=freq
                        indices[pre_pair].add(i)
                    if j<len(word)-1:
                        nex_pair=(word[j],word[j+1])
                        pair_freq[nex_pair]+=freq
                        indices[nex_pair].add(i)
                else:
                    j+=1
        if best_pair in pair_freq:
            del pair_freq[best_pair]
        if best_pair in indices:
            del indices[best_pair]
    for pair in merge_rule:
        vocab[len(vocab)]=pair[0]+pair[1]
    for spec_token in special_tokens:
        vocab[len(vocab)]=spec_token.encode("utf-8")
    return vocab,merge_rule
