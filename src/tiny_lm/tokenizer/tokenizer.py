"""
从 Assignment 1 迁移的字节级 BPE 编解码。
"""

from collections.abc import Iterable,Iterator
from numbers import Integral
from pathlib import Path
from typing import Self
from tiny_lm.tokenizer.rules import iter_pretokens,split_special_tokens
from tiny_lm.tokenizer.serialization import _validate_tokenizer,load_tokenizer,save_tokenizer


class BPETokenizer:
    """
    使用原生词表和有序合并规则编码；构造时复制输入容器。
    """

    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str]|None=None):
        if special_tokens is None:
            special_tokens=[]
        _validate_tokenizer(vocab,merges,special_tokens)
        self.vocab=dict(vocab)
        self.id_to_byte=self.vocab
        self.merges={pair:i for i,pair in enumerate(merges)}
        self.special_tokens=list(special_tokens)
        self.special_to_id={token:256+len(merges)+i for i,token in enumerate(special_tokens)}
        self.byte_to_id={}
        # 普通 token 重复字节取最小 ID，特殊 token 使用独立映射。
        for i in range(256+len(merges)):
            self.byte_to_id.setdefault(self.vocab[i],i)

    @classmethod
    def from_file(cls,path:str|Path)->Self:
        """
        加载经过校验的原生 v1 JSON 文件。
        """
        return cls(*load_tokenizer(path))

    def save(self,path:str|Path)->None:
        """
        保存原生 v1 JSON，不创建父目录或覆盖已有文件。
        """
        merges=sorted(self.merges,key=self.merges.__getitem__)
        save_tokenizer(path,self.vocab,merges,self.special_tokens)

    @staticmethod
    def _check_special_mode(special_mode:str)->None:
        if not isinstance(special_mode,str):
            raise TypeError("special_mode must be a string")
        if special_mode not in ("recognize","ordinary"):
            raise ValueError("special_mode must be 'recognize' or 'ordinary'")

    def encode(self,text:str,*,special_mode:str="recognize")->list[int]:
        """
        默认识别特殊 token；ordinary 将全部输入作为普通文本。
        """
        if not isinstance(text,str):
            raise TypeError("text must be a string")
        self._check_special_mode(special_mode)
        if not text:
            return []
        if special_mode=="ordinary":
            return self._encode_text_segment(text)
        tokens=[]
        for segment,is_special in split_special_tokens(text,self.special_tokens):
            if is_special:
                tokens.append(self.special_to_id[segment])
            else:
                tokens.extend(self._encode_text_segment(segment))
        return tokens

    def _encode_text_segment(self,text:str)->list[int]:
        ids=[]
        for pre_token in iter_pretokens(text,[]):
            byte_parts=[bytes([b]) for b in pre_token.encode("utf-8")]
            while len(byte_parts)>1:
                best_pair=None
                min_rank=float("inf")
                for i in range(len(byte_parts)-1):
                    pair=(byte_parts[i],byte_parts[i+1])
                    if pair in self.merges:
                        rank=self.merges[pair]
                        if rank<min_rank:
                            min_rank=rank
                            best_pair=pair
                if best_pair is None:
                    break
                # 按最小 rank 选择规则，从左到右合并所有非重叠匹配。
                new_byte_parts=[]
                i=0
                while i<len(byte_parts):
                    if i<len(byte_parts)-1 and (byte_parts[i],byte_parts[i+1])==best_pair:
                        new_byte_parts.append(best_pair[0]+best_pair[1])
                        i+=2
                    else:
                        new_byte_parts.append(byte_parts[i])
                        i+=1
                byte_parts=new_byte_parts
            for part in byte_parts:
                ids.append(self.byte_to_id[part])
        return ids

    def decode(self,ids:Iterable[int],*,errors:str="replace")->str:
        """
        拼接全部 token 字节后解码 UTF-8；strict 可检查不完整字节序列。
        """
        if not isinstance(errors,str):
            raise TypeError("errors must be a string")
        if errors not in ("replace","strict"):
            raise ValueError("errors must be 'replace' or 'strict'")
        if isinstance(ids,(str,bytes,bytearray)):
            raise TypeError("ids must be an iterable of integer token IDs")
        try:
            iterator=iter(ids)
        except TypeError as exc:
            raise TypeError("ids must be an iterable of integer token IDs") from exc
        byte_seg=[]
        for i in iterator:
            if isinstance(i,bool) or not isinstance(i,Integral):
                raise TypeError("token IDs must be integers, excluding bool")
            i=int(i)
            if i not in self.id_to_byte:
                raise ValueError(f"unknown token ID: {i}")
            byte_seg.append(self.id_to_byte[i])
        byte_flow=b"".join(byte_seg)
        return byte_flow.decode("utf-8",errors=errors)

    def encode_iterable(self,iterable:Iterable[str],*,special_mode:str="recognize")->Iterator[int]:
        """
        逐条编码独立文本，不跨记录合并，不自动添加分隔符或 EOS。
        不保证任意分块与整段文本编码相同；逐条校验，出错前可能已产出 ID。
        """
        self._check_special_mode(special_mode)
        if isinstance(iterable,(str,bytes,bytearray)):
            raise TypeError("iterable must contain independent text records, not a single string or bytes")
        try:
            iterator=iter(iterable)
        except TypeError as exc:
            raise TypeError("iterable must be an iterable of text records") from exc
        for text in iterator:
            yield from self.encode(text,special_mode=special_mode)
