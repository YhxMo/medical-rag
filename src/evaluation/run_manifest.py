"""Reject cache reuse across changed datasets, contexts or prompts."""
import json
from pathlib import Path


def freeze_run_manifest(directory,manifest):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'run_manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest:
            raise ValueError('Run inputs changed; use a new output directory')
    else:
        # Existing answer files without a manifest must be explicitly migrated.
        if any(p.name!='run_manifest.json' and not p.name.startswith('._') for p in directory.glob('*_answer.json')):
            raise ValueError('Existing answers have no run manifest')
        path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
