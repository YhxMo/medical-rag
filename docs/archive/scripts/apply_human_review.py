"""Import an explicitly exported human decision file; keeps original candidates frozen."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.evaluation.human_review import apply_decisions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('decisions',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'data/evaluation/public-v1-human-reviewed')
    args=p.parse_args()
    source=ROOT/'data/evaluation/public-v1/all_candidates.json'
    rows=json.loads(source.read_text())
    reviewed=apply_decisions(rows,json.loads(args.decisions.read_text()),hashlib.sha256(source.read_bytes()).hexdigest())
    if args.output.exists(): raise ValueError('Output exists; select a new review version')
    args.output.mkdir(parents=True)
    for split in ('dev','eval'):
        (args.output/f'{split}.json').write_text(json.dumps([q for q in reviewed if q['metadata']['split']==split],ensure_ascii=False,indent=2)+'\n')
    summary={'review_kind':'human_source_review_not_medical_expert_certification',
             'total':len(reviewed),'human_accepted':sum(q['reviewed'] for q in reviewed)}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__': main()
