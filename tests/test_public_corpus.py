import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.generate_public_questions import BudgetClient, validate_checks, validate_questions
from scripts.rebuild_public_corpus import select_sources
from src.cli import _build_generator, _build_embedding_provider
from src.config.settings import Settings
from src.embedding.fastembed_provider import FastEmbedProvider
from src.indexer.common import tokenize
from src.schema import EvidenceItem


def test_english_bm25_is_case_insensitive_and_ignores_punctuation():
    assert tokenize('CT, Ultrasound! 3-D') == ['ct', 'ultrasound', '3-d']


def test_fastembed_uses_query_and_passage_encoders():
    provider = FastEmbedProvider()
    class Encoder:
        def passage_embed(self, texts, **kwargs):
            assert texts == ['passage']
            return iter([np.array([1., 0.])])
        def query_embed(self, query):
            assert query == 'query'
            return iter([np.array([0., 1.])])
    provider._model = Encoder()
    assert provider.embed_texts(['passage']) == [[1., 0.]]
    assert provider.embed_query('query') == [0., 1.]
    assert provider.embed_texts([]) == []


def test_deepseek_factory_uses_official_endpoint_and_runtime_key(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'synthetic-test-credential')
    settings = Settings(tmp_path/'unused', {})
    generator = _build_generator('deepseek', settings)
    assert generator.api_base == 'https://api.deepseek.com'
    assert generator.model == 'deepseek-flash'
    assert generator.provider_name == 'deepseek'
    assert 'synthetic-test-credential' not in repr(generator)
    assert isinstance(_build_embedding_provider('fastembed', settings), FastEmbedProvider)


def test_source_sampling_is_deterministic_and_page_disjoint():
    evidence = [EvidenceItem(f'{source}-{page}', 'text', source, 'educational ' * 100, page, page)
                for source in ['a.pdf', 'b.pdf', 'c.pdf'] for page in range(1, 101)]
    plan = select_sources(evidence)
    assert len(plan) == 60
    assert select_sources(evidence) == plan
    dev = {(x['source_file'], x['page']) for x in plan if x['split']=='dev'}
    heldout = {(x['source_file'], x['page']) for x in plan if x['split']=='eval'}
    assert len(dev) == len(heldout) == 30
    assert not dev & heldout


def test_question_validation_rejects_hallucinated_source_mapping():
    with pytest.raises(ValueError):
        validate_questions({'questions':[{'sample_id':'unknown','question':'Some question?', 'answer':'Some valid answer.'}]},
                           [{'sample_id':'s1'}])


def test_automated_review_requires_actual_boolean_flags():
    with pytest.raises(ValueError):
        validate_checks({'checks':[{'sample_id':'s1','grounded':'true','self_contained':True,'ambiguous':False}]}, {'s1'})


def test_budget_blocks_network_before_limit_is_exceeded(tmp_path):
    client = BudgetClient('synthetic-test-credential', tmp_path/'usage.json', limit=0)
    with pytest.raises(RuntimeError, match='Budget'):
        client.call('system', {'sources':[]})
    assert client.report['requests_started'] == 0


def test_http_error_logs_only_status_not_response_or_secret(tmp_path, monkeypatch):
    import httpx
    class FakeClient:
        def __init__(self, **kwargs):
            assert not kwargs['follow_redirects']
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, **kwargs):
            assert url == 'https://api.deepseek.com/chat/completions'
            return SimpleNamespace(status_code=401, text='SYNTHETIC_PRIVATE_BODY')
    monkeypatch.setattr(httpx, 'Client', FakeClient)
    path = tmp_path/'usage.json'
    client = BudgetClient('SYNTHETIC_PRIVATE_KEY', path)
    with pytest.raises(RuntimeError, match='http_401'):
        client.call('SYNTHETIC_PRIVATE_PROMPT', {})
    text = path.read_text()
    assert 'PRIVATE' not in text
    assert json.loads(text)['errors'] == ['http_401']
