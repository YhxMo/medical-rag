"""Bounded neighboring text expansion; never uses question relevance labels."""
from collections import defaultdict
from src.schema import RetrievalHit


class ContextExpander:
    def __init__(self,evidence):
        self.groups=defaultdict(list)
        self.positions={}
        for item in evidence:
            if item.evidence_type=='text' and item.page_start is not None:
                self.groups[item.source_file].append(item)
        for group in self.groups.values():
            group.sort(key=lambda e:(e.page_start,e.metadata.get('char_start',0),e.evidence_id))
            for i,item in enumerate(group):self.positions[item.evidence_id]=(group,i)

    def expand(self,hits,*,max_items=10,max_chars=12000):
        if max_items<1 or max_chars<1:raise ValueError('Positive context budgets required')
        result=[];seen=set();used=0
        def add(item,score,scores):
            nonlocal used
            if item.evidence_id in seen or len(result)>=max_items or used+len(item.content)>max_chars:return
            result.append(RetrievalHit(item,score,len(result)+1,dict(scores)))
            used+=len(item.content);seen.add(item.evidence_id)
        for h in hits:add(h.evidence,h.score,h.scores)
        # Preserve seeds first. Round-robin next then previous so one long page
        # cannot consume the entire budget before other seeds get context.
        for offset in (1,-1):
            for h in hits:
                found=self.positions.get(h.evidence.evidence_id)
                if found is None:continue
                group,i=found;j=i+offset
                if 0<=j<len(group):
                    e=group[j]
                    if abs(e.page_start-h.evidence.page_start)<=1:
                        add(e,0.0,{'context_expansion':1.0})
        return result
