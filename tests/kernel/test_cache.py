import asyncio

from kite.kernel.cache import CachedKernel, CacheStore, cache_key
from kite.kernel.mock import MockKernel
from kite.kernel.types import KernelRequest, QuestionSpec


def noul(text: str) -> QuestionSpec:
    return QuestionSpec(type="noul", instructions=text)


def test_cache_key_is_length_prefixed():
    assert cache_key("ab", "c") != cache_key("a", "bc")
    assert cache_key("a", "b") == cache_key("a", "b")


def test_store_roundtrip_and_export_import(tmp_path):
    store = CacheStore(tmp_path / "a.sqlite")
    assert store.get("k") is None
    store.put("k", backend="mock", model="m", request_json="{}", answer={"x": 1}, latency_ms=3.0)
    assert store.get("k") == {"backend": "mock", "model": "m", "answer": {"x": 1}, "latency_ms": 3.0}
    assert store.stats() == {"mock:m": 1}
    assert store.export_jsonl_gz(tmp_path / "dump.jsonl.gz") == 1

    other = CacheStore(tmp_path / "b.sqlite")
    assert other.import_jsonl_gz(tmp_path / "dump.jsonl.gz") == 1
    assert other.get("k")["answer"] == {"x": 1}


def test_question_granularity_second_call_is_free(tmp_path):
    inner = MockKernel()
    kernel = CachedKernel(inner, CacheStore(tmp_path / "c.sqlite"), granularity="question")
    request = KernelRequest(state={"a": 1}, questions={"q1": noul("one"), "q2": noul("two")})
    first = asyncio.run(kernel.evaluate(request))
    second = asyncio.run(kernel.evaluate(request))
    assert len(inner.seen) == 1
    assert first.cached == {"q1": False, "q2": False}
    assert second.cached == {"q1": True, "q2": True}
    assert second.answers == first.answers
    assert (kernel.hits, kernel.misses) == (2, 2)


def test_question_granularity_sends_only_missing_questions(tmp_path):
    inner = MockKernel()
    kernel = CachedKernel(inner, CacheStore(tmp_path / "c.sqlite"), granularity="question")
    asyncio.run(kernel.evaluate(KernelRequest(state="s", questions={"q1": noul("one")})))
    mixed = asyncio.run(kernel.evaluate(KernelRequest(state="s", questions={"q1": noul("one"), "q2": noul("two")})))
    assert list(inner.seen[1].questions) == ["q2"]
    assert mixed.cached == {"q1": True, "q2": False}
    assert list(mixed.answers) == ["q1", "q2"]


def test_question_key_does_not_depend_on_the_question_name(tmp_path):
    inner = MockKernel()
    kernel = CachedKernel(inner, CacheStore(tmp_path / "c.sqlite"), granularity="question")
    asyncio.run(kernel.evaluate(KernelRequest(state="s", questions={"first_name": noul("same")})))
    again = asyncio.run(kernel.evaluate(KernelRequest(state="s", questions={"other_name": noul("same")})))
    assert len(inner.seen) == 1
    assert again.cached == {"other_name": True}


def test_state_field_order_changes_the_key(tmp_path):
    inner = MockKernel()
    kernel = CachedKernel(inner, CacheStore(tmp_path / "c.sqlite"), granularity="question")
    asyncio.run(kernel.evaluate(KernelRequest(state={"a": 1, "b": 2}, questions={"q": noul("x")})))
    asyncio.run(kernel.evaluate(KernelRequest(state={"b": 2, "a": 1}, questions={"q": noul("x")})))
    assert len(inner.seen) == 2


def test_salt_forces_a_fresh_evaluation(tmp_path):
    inner = MockKernel()
    store = CacheStore(tmp_path / "c.sqlite")
    request = KernelRequest(state="s", questions={"q": noul("x")})
    asyncio.run(CachedKernel(inner, store, granularity="question").evaluate(request))
    asyncio.run(CachedKernel(inner, store, granularity="question", salt="repeat-1").evaluate(request))
    assert len(inner.seen) == 2


def test_request_granularity_misses_when_any_question_changes(tmp_path):
    inner = MockKernel()
    kernel = CachedKernel(inner, CacheStore(tmp_path / "c.sqlite"), granularity="request")
    asyncio.run(kernel.evaluate(KernelRequest(state="s", questions={"q1": noul("one")})))
    asyncio.run(kernel.evaluate(KernelRequest(state="s", questions={"q1": noul("one")})))
    asyncio.run(kernel.evaluate(KernelRequest(state="s", questions={"q1": noul("one"), "q2": noul("two")})))
    assert [list(r.questions) for r in inner.seen] == [["q1"], ["q1", "q2"]]


def test_on_lookup_callback_receives_counts(tmp_path):
    seen = []
    kernel = CachedKernel(MockKernel(), CacheStore(tmp_path / "c.sqlite"), granularity="question", on_lookup=lambda h, m: seen.append((h, m)))
    request = KernelRequest(state="s", questions={"q": noul("x")})
    asyncio.run(kernel.evaluate(request))
    asyncio.run(kernel.evaluate(request))
    assert seen == [(0, 1), (1, 0)]
