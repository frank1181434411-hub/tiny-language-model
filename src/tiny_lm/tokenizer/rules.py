"""
BPE 与 Tokenizer 共用的预分词规则。
"""

from collections.abc import Iterator
import re

PRETOKENIZER="gpt2-regex-v1"
PRETOKEN_PATTERN=r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def validate_special_tokens(special_tokens:list[str])->None:
    """
    检查特殊 token，保留调用方给定的 ID 顺序。
    """
    if not isinstance(special_tokens,list):
        raise TypeError("special_tokens must be a list of strings")
    seen=set()
    for token in special_tokens:
        if not isinstance(token,str):
            raise TypeError("special_tokens must contain only strings")
        if not token:
            raise ValueError("special_tokens must not contain empty strings")
        if token in seen:
            raise ValueError("special_tokens must not contain duplicates")
        try:
            token.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("special_tokens must contain valid UTF-8 text") from exc
        seen.add(token)


def split_special_tokens(text:str,special_tokens:list[str])->Iterator[tuple[str,bool]]:
    """
    按最左位置、最长匹配切分，返回文本与是否为特殊 token。
    """
    if not isinstance(text,str):
        raise TypeError("text must be a string")
    validate_special_tokens(special_tokens)
    if not special_tokens:
        if text:
            yield text,False
        return
    special_regex="|".join(re.escape(token) for token in sorted(special_tokens,key=len,reverse=True))
    start=0
    for match in re.finditer(special_regex,text):
        if match.start()>start:
            yield text[start:match.start()],False
        yield match.group(),True
        start=match.end()
    if start<len(text):
        yield text[start:],False


def iter_pretokens(text:str,special_tokens:list[str])->Iterator[str]:
    """
    只返回普通文本的预分词，不跨特殊 token 边界合并。
    """
    try:
        import regex
    except ModuleNotFoundError as exc:
        raise ImportError('BPE training and text encoding require regex; install tiny-language-model[tokenizer]') from exc
    pretoken_pat=regex.compile(PRETOKEN_PATTERN)
    for segment,is_special in split_special_tokens(text,special_tokens):
        if not is_special:
            for match in pretoken_pat.finditer(segment):
                yield match.group()
