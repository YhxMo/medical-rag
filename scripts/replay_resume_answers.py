"""Clearly labeled historical answer replay; no retrieval or model requests."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    row = json.loads(args.record.read_text())
    print("HISTORICAL REPLAY / 历史回答回放：不是本次实时生成或性能测量。")
    result = row.get("result", row)
    print(result.get("answer", "No saved answer"))
    print(
        json.dumps(
            {
                "question_id": row.get("question_id"),
                "status": result.get("status"),
                "source_count": len(result.get("sources", [])),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
