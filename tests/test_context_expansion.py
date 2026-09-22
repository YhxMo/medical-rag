import pytest
from src.schema import EvidenceItem,RetrievalHit
from src.retrieval.context import ContextExpander


def item(i,page=1,source='book',start=0):return EvidenceItem(str(i),'text',source,'a'*10,page,page,{'char_start':start})
def hit(e):return RetrievalHit(e,1,1,{})

def test_neighbors_are_ordered_and_do_not_cross_books():
    a,b,c=item(1,start=0),item(2,start=10),item(3,source='other')
    out=ContextExpander([c,b,a]).expand([hit(a)])
    assert [h.evidence.evidence_id for h in out]==['1','2']
    assert out[1].scores=={'context_expansion':1.0}

def test_duplicate_seeds_neighbors_and_budget():
    a,b,c=item(1),item(2,start=10),item(3,start=20)
    out=ContextExpander([a,b,c]).expand([hit(b),hit(b)],max_chars=20)
    assert [h.evidence.evidence_id for h in out]==['2','3']
    assert [h.rank for h in out]==[1,2]

def test_missing_pages_not_bridged():
    a,b=item(1,1),item(2,4)
    assert len(ContextExpander([a,b]).expand([hit(a)]))==1

def test_empty_input_stays_empty():
    assert ContextExpander([item(1)]).expand([])==[]

def test_invalid_budget_rejected():
    with pytest.raises(ValueError):ContextExpander([]).expand([],max_items=0)
