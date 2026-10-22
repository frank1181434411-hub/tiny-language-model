import builtins
import json
import random
import pytest
from tiny_lm.tokenizer.bpe import train_bpe
from tiny_lm.tokenizer.tokenizer import BPETokenizer

pytest.importorskip("regex")


def tokenizer_data(merges=(),specials=()):
    tokens=[bytes([i]) for i in range(256)]+[left+right for left,right in merges]+[token.encode("utf-8") for token in specials]
    return dict(enumerate(tokens)),list(merges),list(specials)


def reference_encode_word(word,merges,vocab):
    # 独立参考：按优先级逐条应用规则，在整数序列上替换。
    ids=list(word.encode("utf-8"))
    for rank,(left,right) in enumerate(merges):
        left_id=min(i for i,token in vocab.items() if token==left)
        right_id=min(i for i,token in vocab.items() if token==right)
        i=0
        while i+1<len(ids):
            if ids[i]==left_id and ids[i+1]==right_id:
                ids[i:i+2]=[256+rank]
            i+=1
    return ids


@pytest.mark.parametrize("text",["","hello world","你好，世界！","🙂👩\u200d💻","e\u0301é","a\r\nb\n\t  ","\x00\x01\x7f","<S><S>","𐐀中文abc123!","   "])
def test_base_vocabulary_exact_utf8_round_trip(text):
    tokenizer=BPETokenizer(*tokenizer_data())
    ids=tokenizer.encode(text)
    assert ids==list(text.encode("utf-8"))
    assert tokenizer.decode(ids,errors="strict")==text


@pytest.mark.parametrize("text,expected",[("ab",[256]),("abab",[257]),("ababa",[257,97]),("aba",[258]),("a b",[97,32,98])])
def test_known_merges(text,expected):
    tokenizer=BPETokenizer(*tokenizer_data([(b"a",b"b"),(b"ab",b"ab"),(b"ab",b"a")]))
    assert tokenizer.encode(text)==expected
    assert tokenizer.decode(expected)==text


def test_rank_priority_over_position_or_frequency():
    tokenizer=BPETokenizer(*tokenizer_data([(b"b",b"c"),(b"a",b"b")]))
    assert tokenizer.encode("abcababc")==[97,256,257,97,256]


@pytest.mark.parametrize("pair,text",[((b"a",b"1"),"a1"),((b"a",b"!"),"a!"),((b"a",b" "),"a "),((b"a",b"\n"),"a\n")])
def test_no_merge_across_pretoken_boundaries(pair,text):
    tokenizer=BPETokenizer(*tokenizer_data([pair]))
    assert tokenizer.encode(text)==list(text.encode())


@pytest.mark.parametrize("seed",range(25))
def test_trained_rules_match_independent_encoder(tmp_path,seed):
    rng=random.Random(seed)
    words=["".join(rng.choices("abc你好",k=rng.randint(1,16))) for _ in range(35)]
    path=tmp_path/"corpus.txt"
    path.write_bytes("\n".join(words).encode("utf-8"))
    vocab,merges=train_bpe(path,290,[])
    tokenizer=BPETokenizer(vocab,merges)
    unseen=["".join(rng.choices("abc你好",k=rng.randint(1,25))) for _ in range(20)]
    expected=[]
    for i,word in enumerate(words+unseen):
        if i:
            expected.append(10)
        expected.extend(reference_encode_word(word,merges,vocab))
    text="\n".join(words+unseen)
    assert tokenizer.encode(text)==expected
    assert tokenizer.decode(expected,errors="strict")==text


@pytest.mark.parametrize("specials",[["<X>","<X>tail"],["<X>tail","<X>"]])
def test_longest_adjacent_specials(specials):
    tokenizer=BPETokenizer(*tokenizer_data(specials=specials))
    ids=tokenizer.encode("a<X>tail<X>b")
    assert ids==[97,256+specials.index("<X>tail"),256+specials.index("<X>"),98]
    assert tokenizer.decode(ids)=="a<X>tail<X>b"


def test_special_literals_and_ordinary_mode():
    tokenizer=BPETokenizer(*tokenizer_data(specials=["[.*]","a","<结束>"]))
    text="a[.*]<结束><unknown>"
    assert tokenizer.encode(text)==[257,256,258]+list(b"<unknown>")
    ids=tokenizer.encode(text,special_mode="ordinary")
    assert ids==list(text.encode("utf-8"))
    assert tokenizer.decode(ids)==text
    assert tokenizer.byte_to_id[b"a"]==97
    assert tokenizer.special_to_id["a"]==257


def test_same_bytes_for_normal_merge_and_special():
    tokenizer=BPETokenizer(*tokenizer_data([(b"a",b"b")],["ab"]))
    assert tokenizer.encode("ab")==[257]
    assert tokenizer.encode("ab",special_mode="ordinary")==[256]
    assert tokenizer.decode([256,257])=="abab"


def test_special_boundary_prevents_merging():
    tokenizer=BPETokenizer(*tokenizer_data([(b"a",b"b")],["<S>"]))
    assert tokenizer.encode("a<S>b")==[97,257,98]


def test_duplicate_normal_bytes_choose_smallest_id():
    data=tokenizer_data([(b"a",b"b"),(b"b",b"c"),(b"ab",b"c"),(b"a",b"bc")])
    vocab,merges,specials=data
    tokenizer=BPETokenizer(dict(reversed(list(vocab.items()))),merges,specials)
    assert tokenizer.encode("abc")==[258]
    assert tokenizer.byte_to_id[b"abc"]==258
    assert tokenizer.decode([258,259])=="abcabc"


def test_constructor_copies_inputs(tmp_path):
    vocab,merges,specials=tokenizer_data([(b"a",b"b")],["<S>"])
    tokenizer=BPETokenizer(vocab,merges,specials)
    vocab.clear()
    merges.clear()
    specials.append("a")
    assert tokenizer.encode("ab<S>")==[256,257]
    assert tokenizer.decode([97,256,257])=="aab<S>"
    path=tmp_path/"tokenizer.json"
    tokenizer.save(path)
    assert BPETokenizer.from_file(path).encode("ab<S>")==[256,257]


@pytest.mark.parametrize("case",["vocab_type","missing_byte","wrong_base","gap","bool_id","wrong_merge","duplicate_merge","future_merge","special_value","special_duplicate","special_missing","special_type"])
def test_invalid_constructor(case):
    vocab,merges,specials=tokenizer_data([(b"a",b"b"),(b"ab",b"c")],["<S>"])
    if case=="vocab_type":
        vocab=[]
    elif case=="missing_byte":
        del vocab[0]
    elif case=="wrong_base":
        vocab[0]=b"x"
    elif case=="gap":
        vocab[999]=vocab.pop(258)
    elif case=="bool_id":
        vocab[True]=vocab.pop(1)
    elif case=="wrong_merge":
        vocab[256]=b"wrong"
    elif case=="duplicate_merge":
        merges[1]=merges[0]
    elif case=="future_merge":
        merges.reverse()
    elif case=="special_value":
        specials[0]=""
    elif case=="special_duplicate":
        specials.append("<S>")
    elif case=="special_missing":
        specials=None
    elif case=="special_type":
        specials="<S>"
    with pytest.raises((ValueError,TypeError)):
        BPETokenizer(vocab,merges,specials)


@pytest.mark.parametrize("text",[None,False,0,1.5,b"hello",[],{}])
def test_encode_rejects_non_strings(text):
    with pytest.raises(TypeError,match="text must be a string"):
        BPETokenizer(*tokenizer_data()).encode(text)


def test_encode_rejects_surrogates():
    with pytest.raises(UnicodeEncodeError):
        BPETokenizer(*tokenizer_data()).encode("a\ud800")


@pytest.mark.parametrize("mode,error",[(None,TypeError),(True,TypeError),([],TypeError),("",ValueError),("all",ValueError)])
def test_invalid_special_mode_even_for_empty_input(mode,error):
    tokenizer=BPETokenizer(*tokenizer_data())
    with pytest.raises(error,match="special_mode"):
        tokenizer.encode("",special_mode=mode)
    with pytest.raises(error,match="special_mode"):
        list(tokenizer.encode_iterable([],special_mode=mode))


@pytest.mark.parametrize("ids,error",[(None,TypeError),(1,TypeError),("a",TypeError),(b"a",TypeError),(bytearray(b"a"),TypeError),([True],TypeError),([97.0],TypeError),(["97"],TypeError),([None],TypeError),([-1],ValueError),([256],ValueError),([10**100],ValueError)])
def test_decode_rejects_invalid_ids(ids,error):
    with pytest.raises(error,match="(ids must|token ID)"):
        BPETokenizer(*tokenizer_data()).decode(ids)


def test_decode_joins_utf8_before_decoding():
    tokenizer=BPETokenizer(*tokenizer_data())
    assert tokenizer.decode(iter("你".encode()),errors="strict")=="你"
    assert tokenizer.decode([])==""
    assert tokenizer.decode([0xff,0xe4])=="\ufffd\ufffd"
    with pytest.raises(UnicodeDecodeError):
        tokenizer.decode([0xe4],errors="strict")


def test_decode_numpy_integer_ids():
    np=pytest.importorskip("numpy")
    tokenizer=BPETokenizer(*tokenizer_data())
    assert tokenizer.decode(np.array([97,98],dtype=np.uint16))=="ab"
    with pytest.raises(TypeError,match="token IDs"):
        tokenizer.decode(np.array([True]))


@pytest.mark.parametrize("errors,error",[(None,TypeError),(True,TypeError),("ignore",ValueError),("",ValueError)])
def test_invalid_decode_errors(errors,error):
    with pytest.raises(error,match="errors"):
        BPETokenizer(*tokenizer_data()).decode([],errors=errors)


def test_iterable_is_lazy_and_does_not_join_records():
    tokenizer=BPETokenizer(*tokenizer_data([(b"a",b"b")],["<S>"]))
    consumed=[]
    def records():
        for text in ("ab","","a","b","<S>"):
            consumed.append(text)
            yield text
    ids=tokenizer.encode_iterable(records())
    assert consumed==[]
    assert next(ids)==256
    assert consumed==["ab"]
    assert list(ids)==[97,98,257]
    assert tokenizer.encode("ab")==[256]
    assert list(tokenizer.encode_iterable(["a","b"]))==[97,98]
    assert list(tokenizer.encode_iterable(["<S>"],special_mode="ordinary"))==list(b"<S>")


def test_iterable_special_split_and_empty_records():
    tokenizer=BPETokenizer(*tokenizer_data(specials=["<S>"]))
    assert list(tokenizer.encode_iterable(["<S",">"]))==list(b"<S>")
    assert tokenizer.encode("<S>")==[256]
    assert list(tokenizer.encode_iterable(["","",""]))==[]


@pytest.mark.parametrize("records",[None,1,"abc",b"abc",bytearray(b"abc")])
def test_iterable_rejects_invalid_container(records):
    with pytest.raises(TypeError,match="iterable must"):
        list(BPETokenizer(*tokenizer_data()).encode_iterable(records))


def test_iterable_reports_bad_record_after_previous_output():
    ids=BPETokenizer(*tokenizer_data()).encode_iterable(["a",None])
    assert next(ids)==97
    with pytest.raises(TypeError,match="text must be a string"):
        next(ids)


def test_train_save_load_encode_workflow(tmp_path):
    corpus=tmp_path/"corpus.txt"
    text="Hello 你好🙂\r\n<S>abababa"
    corpus.write_bytes(text.encode("utf-8"))
    vocab,merges=train_bpe(corpus,300,["<S>"])
    tokenizer=BPETokenizer(vocab,merges,["<S>"])
    ids=tokenizer.encode(text)
    path=tmp_path/"tokenizer.json"
    tokenizer.save(path)
    restored=BPETokenizer.from_file(path)
    assert restored.encode(text)==ids
    assert restored.decode(ids,errors="strict")==text
    assert list(restored.encode_iterable([text,"",text]))==ids+ids
    before=path.read_bytes()
    with pytest.raises(FileExistsError):
        restored.save(path)
    assert path.read_bytes()==before


def test_from_file_checks_metadata(tmp_path):
    path=tmp_path/"tokenizer.json"
    BPETokenizer(*tokenizer_data()).save(path)
    payload=json.loads(path.read_text(encoding="utf-8"))
    payload["version"]=99
    path.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError,match="unsupported tokenizer version"):
        BPETokenizer.from_file(path)


def test_missing_dependency_only_blocks_text_processing(monkeypatch,tmp_path):
    original=builtins.__import__
    def without_regex(name,*args,**kwargs):
        if name=="regex":
            raise ModuleNotFoundError("No module named 'regex'")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",without_regex)
    tokenizer=BPETokenizer(*tokenizer_data())
    assert tokenizer.decode([97])=="a"
    path=tmp_path/"tokenizer.json"
    tokenizer.save(path)
    tokenizer=BPETokenizer.from_file(path)
    with pytest.raises(ImportError,match=r"tiny-language-model\[tokenizer\]"):
        tokenizer.encode("a")
