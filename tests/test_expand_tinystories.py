"""
验证数据抽样边界和评估集泄漏防护，不访问网络。
"""
from pathlib import Path
import runpy
import pytest

MODULE=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts"/"expand_tinystories.py"))


def test_partial_utf8_edges_are_discarded_but_internal_paragraphs_survive():
    raw=b"\xa0partial<|endoftext|>\r\n Lily smiled.\r\n\r\nShe ran. \r\n<|endoftext|>tail\xe4"
    assert MODULE["complete_stories"](raw)==["Lily smiled.\n\nShe ran."]


def test_full_delimited_story_required():
    assert MODULE["complete_stories"](b"unfinished story")==[]
    with pytest.raises(UnicodeDecodeError):
        MODULE["complete_stories"](b"edge<|endoftext|>bad\xff<|endoftext|>edge")


def test_overlap_detects_small_edits_and_normalized_short_text():
    index=MODULE["OverlapIndex"]()
    original=" ".join(f"word{i}" for i in range(100))
    assert not index.check(original,add=True)
    assert index.check(original.replace("word50","changed"))
    assert not index.check("An unrelated little girl played with her dog.")
    index.check("Hi, Lily!",add=True)
    assert index.check("HI lily")


def test_all_protected_texts_remain_indexed_even_when_similar():
    index=MODULE["OverlapIndex"]()
    original=" ".join(f"word{i}" for i in range(100))
    second=original.replace("word20","changed20")
    third=second.replace("word40","changed40")
    index.check(original,add=True)
    assert index.check(second,add=True)
    assert index.check(third)
    assert len(index.sizes)==2


def test_full_download_response_rejected_before_body_read(tmp_path,monkeypatch):
    class Response:
        status=200
        headers={}

        def __enter__(self):
            return self

        def __exit__(self,*args):
            return False

        def read(self,*args):
            raise AssertionError("full response body must not be downloaded")

    monkeypatch.setattr(MODULE["urllib"].request,"urlopen",lambda *args,**kwargs:Response())
    with pytest.raises(ValueError,match="refusing full download"):
        MODULE["fetch_candidates"]("train",tmp_path,42,1,1024)
    assert list(tmp_path.iterdir())==[]
