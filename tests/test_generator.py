"""Offline tests for the generation pipeline using a fake retriever and a fake LLM client."""

import json

import pytest

from generation.config import GenerationConfig
from generation.context import format_context, format_source
from generation.generator import AnswerGenerator
from generation.prompts import SYSTEM_PROMPTS, build_user_prompt, get_system_prompt
from generation.schemas import LLMResponse
from retriever.hybrid_retriever import SearchResult


def _chunk(i: int, **extra) -> dict:
    chunk = {
        "chunk_id": f"AAPL_10-K_2025_c{i:04d}",
        "doc_id": "AAPL_10-K_2025",
        "section": "Item 7",
        "section_title": "MD&A",
        "type": "text",
        "element_orders": [i],
        "text": f"chunk text {i}",
        "breadcrumb": "AAPL 10-K 2025 > Item 7",
    }
    chunk.update(extra)
    return chunk


class FakeLLM:
    provider = "fake"
    model = "fake-model"
    temperature = 0.0
    max_tokens = 100

    def __init__(self, content: str | Exception):
        self.content = content
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self.content, Exception):
            raise self.content
        return LLMResponse(self.content, self.model, 11, 7, 0.25)


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls: list[dict] = []

    def search(self, query, k=None, doc_id=None, **_):
        self.calls.append({"query": query, "k": k, "doc_id": doc_id})
        return self.results[:k]


def _config(**overrides) -> GenerationConfig:
    return GenerationConfig(provider="ollama", model="m", context_k=2, **overrides)


def _generator(llm: FakeLLM, retriever: FakeRetriever | None = None) -> AnswerGenerator:
    return AnswerGenerator(_config(), retriever=retriever, llm_client=llm)


GOOD = json.dumps(
    {"answer": "$391 billion", "explanation": "e", "citations": ["S1"], "calculation": None, "insufficient_context": False}
)


class TestFormatContext:
    def test_source_ids_are_one_based_and_ordered(self):
        context, sources = format_context([SearchResult(0.9, _chunk(0)), SearchResult(0.5, _chunk(1))])
        assert [s["source_id"] for s in sources] == ["S1", "S2"]
        assert context.index('id="S1"') < context.index('id="S2"')

    def test_accepts_raw_dicts_without_score(self):
        _, sources = format_context([_chunk(0)])
        assert sources[0]["score"] is None and sources[0]["text"] == "chunk text 0"

    def test_search_result_score_is_recorded(self):
        _, sources = format_context([SearchResult(0.75, _chunk(0))])
        assert sources[0]["score"] == 0.75

    def test_source_block_attributes(self):
        block = format_source(3, _chunk(5, type="table", table_id="T1", units="millions"))
        assert block.startswith('<source id="S3" doc="AAPL_10-K_2025" section="MD&A" type="table"')
        assert 'elements="[5]"' in block and 'table_id="T1"' in block and 'units="millions"' in block
        assert block.endswith("chunk text 5\n</source>")

    def test_section_falls_back_to_unknown(self):
        assert 'section="Unknown"' in format_source(1, {"doc_id": "D", "text": "t"})

    def test_empty_results(self):
        assert format_context([]) == ("", [])


class TestPrompts:
    def test_all_versions_resolve(self):
        for version, prompt in SYSTEM_PROMPTS.items():
            assert get_system_prompt(version) == prompt and "insufficient_context" in prompt

    def test_unknown_version_lists_supported(self):
        with pytest.raises(ValueError, match="v2_complete_precise"):
            get_system_prompt("nope")

    def test_user_prompt_contains_question_and_sources(self):
        prompt = build_user_prompt("What is X?", "<source id=\"S1\">t</source>")
        assert "What is X?" in prompt and '<source id="S1">' in prompt


class TestGenerate:
    def test_happy_path(self):
        llm = FakeLLM(GOOD)
        result = _generator(llm).generate("Q?", [SearchResult(0.9, _chunk(0))], doc_id="AAPL_10-K_2025")
        assert result.answer.answer == "$391 billion" and result.answer.is_valid
        assert result.doc_id == "AAPL_10-K_2025" and result.model == "fake-model"
        assert (result.prompt_tokens, result.completion_tokens, result.latency_seconds) == (11, 7, 0.25)
        assert result.context_k == 1 and result.raw_response == GOOD
        assert [s["source_id"] for s in result.sources] == ["S1"]

    def test_prompt_contains_question_and_context(self):
        llm = FakeLLM(GOOD)
        _generator(llm).generate("What is X?", [SearchResult(0.9, _chunk(0))])
        system_prompt, user_prompt = llm.calls[0]
        assert system_prompt == get_system_prompt("v2_complete_precise")
        assert "What is X?" in user_prompt and "chunk text 0" in user_prompt

    def test_no_chunks_abstains_without_calling_llm(self):
        llm = FakeLLM(GOOD)
        result = _generator(llm).generate("Q?", [])
        assert llm.calls == []
        assert result.answer.insufficient_context and result.answer.answer is None
        assert result.sources == [] and result.context_k == 0

    def test_invalid_citation_marks_answer_invalid(self):
        bad = json.dumps({"answer": "a", "explanation": "e", "citations": ["S7"]})
        result = _generator(FakeLLM(bad)).generate("Q?", [SearchResult(0.9, _chunk(0))])
        assert not result.answer.is_valid
        assert any("S7" in e for e in result.answer.validation_errors)

    def test_unparseable_llm_output_is_invalid_not_an_exception(self):
        result = _generator(FakeLLM("sorry, I cannot")).generate("Q?", [SearchResult(0.9, _chunk(0))])
        assert not result.answer.is_valid and result.answer.answer is None

    def test_llm_errors_propagate(self):
        generator = _generator(FakeLLM(RuntimeError("provider down")))
        with pytest.raises(RuntimeError, match="provider down"):
            generator.generate("Q?", [SearchResult(0.9, _chunk(0))])

    def test_result_to_dict_is_json_serializable(self):
        result = _generator(FakeLLM(GOOD)).generate("Q?", [SearchResult(0.9, _chunk(0))])
        assert json.loads(json.dumps(result.to_dict()))["answer"]["answer"] == "$391 billion"


class TestAnswer:
    def test_retrieve_uses_filter_and_default_context_k(self):
        retriever = FakeRetriever([SearchResult(0.9, _chunk(i)) for i in range(5)])
        result = _generator(FakeLLM(GOOD), retriever).answer("Q?", doc_id="AAPL_10-K_2025")
        assert retriever.calls == [{"query": "Q?", "k": 2, "doc_id": "AAPL_10-K_2025"}]
        assert result.context_k == 2

    def test_context_k_override(self):
        retriever = FakeRetriever([SearchResult(0.9, _chunk(i)) for i in range(5)])
        _generator(FakeLLM(GOOD), retriever).answer("Q?", context_k=4)
        assert retriever.calls[0]["k"] == 4

    def test_empty_retrieval_abstains(self):
        llm = FakeLLM(GOOD)
        result = _generator(llm, FakeRetriever([])).answer("Q?", doc_id="AAPL_10-K_2025")
        assert llm.calls == [] and result.answer.insufficient_context


class TestRetrieverSelection:
    def test_injected_retriever_is_not_replaced(self):
        retriever = FakeRetriever([])
        assert _generator(FakeLLM(GOOD), retriever).retriever is retriever

    def test_numpy_store_builds_hybrid_retriever(self, monkeypatch):
        import retriever.hybrid_retriever as hybrid
        from config.settings import settings

        monkeypatch.setattr(settings, "VECTOR_STORE", "numpy")
        built = {}

        class Stub:
            chunks = [1, 2, 3]

            def __init__(self, **kwargs):
                built.update(kwargs)

        monkeypatch.setattr(hybrid, "HybridRetrieverWithReranker", Stub)
        generator = AnswerGenerator(_config(), llm_client=FakeLLM(GOOD), root="/project")
        assert isinstance(generator.retriever, Stub)
        assert built["rerank_k"] == 2 and built["run_directory"].replace("\\", "/").startswith("/project")

    def test_qdrant_store_builds_qdrant_retriever(self, monkeypatch):
        import retriever.qdrant_retriever as qdrant
        from config.settings import settings

        monkeypatch.setattr(settings, "VECTOR_STORE", "qdrant")
        built = {}

        class Stub:
            chunk_count = 3

            def __init__(self, **kwargs):
                built.update(kwargs)

        monkeypatch.setattr(qdrant, "QdrantRetrieverWithReranker", Stub)
        generator = AnswerGenerator(_config(), llm_client=FakeLLM(GOOD))
        assert isinstance(generator.retriever, Stub) and built["rerank_k"] == 2


class TestAnswerBatch:
    def test_success_and_error_rows(self):
        class Flaky(FakeRetriever):
            def search(self, query, k=None, doc_id=None, **_):
                if "boom" in query:
                    raise RuntimeError("retriever failed")
                return super().search(query, k=k, doc_id=doc_id)

        generator = _generator(FakeLLM(GOOD), Flaky([SearchResult(0.9, _chunk(0))]))
        rows = generator.answer_batch(
            [
                {"id": "q1", "query": "fine", "doc_id": "AAPL_10-K_2025", "expected_answer": "x"},
                {"id": "q2", "query": "boom"},
            ]
        )
        assert [r["status"] for r in rows] == ["success", "error"]
        assert rows[0]["generated_answer"] == "$391 billion" and rows[0]["is_valid"]
        assert rows[1]["error"] == "retriever failed" and rows[1]["id"] == "q2"
