"""Evaluate a finished task: load its final_artifact + source texts, score, write a report.

Usage (real run, ml/GigaChat env):
    python -m src.evaluation.run <task_id> --source data/demo/transcript.txt --out data/demo/eval-report.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.db.repos import TaskRepo
from src.db.session import create_engine_and_session
from src.evaluation.intrinsic import (
    citations_intrinsics,
    quiz_intrinsics,
    summary_intrinsics,
    terms_intrinsics,
)
from src.evaluation.judge import judge_quiz, judge_summary
from src.evaluation.report import build_report
from src.main import _build_llm


def _summary_text(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""
    return "\n".join(str(s.get("text", "")) for s in summary.get("sections", []) if isinstance(s, dict))


def _quiz_text(quiz: list[dict[str, Any]]) -> str:
    return "\n".join(str(q.get("question", "")) for q in quiz)


async def _load_artifact(task_id: str) -> dict[str, Any]:
    settings = get_settings()
    engine, session_factory = create_engine_and_session(settings.database_url)
    try:
        async with session_factory() as session:
            row = await TaskRepo(session).get(task_id)
            if row is None or row.final_artifact is None:
                raise SystemExit(f"task {task_id} has no final_artifact")
            return dict(row.final_artifact)
    finally:
        await engine.dispose()


async def _run(task_id: str, source_path: str, out_path: str) -> None:
    artifact = await _load_artifact(task_id)
    result = artifact.get("result", {})
    source_text = Path(source_path).read_text(encoding="utf-8") if source_path and Path(source_path).exists() else ""
    intrinsic = {
        "summary": summary_intrinsics(result.get("summary"), len(source_text)),
        "terms": terms_intrinsics(result.get("terms", []), source_text),
        "quiz": quiz_intrinsics(result.get("quiz", [])),
        "citations": citations_intrinsics(result.get("citations", [])),
    }
    llm = _build_llm(get_settings())
    summary_text = _summary_text(result.get("summary"))
    summary_judge = await judge_summary(llm, summary_text=summary_text, source_excerpt=source_text)
    quiz_list = result.get("quiz", [])
    quiz_judge = (
        await judge_quiz(llm, quiz_text=_quiz_text(quiz_list), summary_text=summary_text) if quiz_list else None
    )
    judge = {
        "summary": summary_judge.model_dump() if summary_judge else None,
        "quiz": quiz_judge.model_dump() if quiz_judge else None,
    }
    report = build_report(task_id=task_id, timing=artifact.get("stats", {}), intrinsic=intrinsic, judge=judge)
    Path(out_path).write_text(report.markdown, encoding="utf-8")
    json_text = json.dumps(report.data, ensure_ascii=False, indent=2)
    Path(out_path).with_suffix(".json").write_text(json_text, encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a finished task's output quality.")
    parser.add_argument("task_id")
    parser.add_argument("--source", default="")
    parser.add_argument("--out", default="data/demo/eval-report.md")
    args = parser.parse_args()
    asyncio.run(_run(args.task_id, args.source, args.out))


if __name__ == "__main__":
    main()
