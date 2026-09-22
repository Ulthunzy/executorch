"""Collect supplied run directories without treating missing/review-pending scores as zero."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

if __package__ in (None, "", "reporting"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_validation.common.audit import audit_run
from model_validation.common.output import atomic_directory


def collect(directory):
    """Expose scores only after auditing their saved evidence."""
    directory = Path(directory)
    path = directory / "summary.json"
    row = {
        "run": str(directory),
        "status": "missing",
        "metric": None,
        "ppl": None,
        "targets": None,
        "planned": None,
        "attempted": None,
        "reviewed": None,
        "normal_rate": None,
        "identity_correct_rate": None,
        "eos": None,
        "repetition_flags": None,
        "error": None,
    }
    if not path.is_file():
        return row
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError("summary.json must contain an object")
        known = summary.get("schema") in (
            "executorch.ppl.summary.v1",
            "executorch.business.summary.v1",
        ) or (
            "planned" in summary
            and summary.get("schema") is None
            and summary.get("schema_version", 1) == 1
        )
        if known:
            audit_run(directory)
    except (OSError, ValueError) as error:
        row.update(status="invalid_evidence", error=str(error))
        return row
    if not known:
        if summary.get("status") == "planned_only":
            row.update(status="planned_only", metric="ppl_plan")
            return row
        row.update(
            status="unverified_or_incomplete",
            metric="ppl_unverified" if "ppl" in summary else "generation_unverified",
        )
        return row
    if summary.get("schema") == "executorch.ppl.summary.v1":
        ppl = summary["ppl"]
        count = summary.get("count", summary.get("targets"))
        valid = (
            summary.get("complete") is True
            and isinstance(ppl, (int, float))
            and math.isfinite(ppl)
            and ppl > 0
            and isinstance(count, int)
            and count > 0
        )
        # Historical artifacts without eligibility evidence are never promoted.
        eligible = summary.get("standard_ppl_eligible") is True
        row.update(
            metric="standard_ppl" if eligible else "ppl_unverified",
            status="complete" if valid and eligible else "unverified_or_incomplete",
            ppl=ppl if valid else None,
            targets=count,
        )
    else:
        for key in (
            "planned",
            "attempted",
            "reviewed",
            "normal_rate",
            "identity_correct_rate",
            "eos",
            "repetition_flags",
        ):
            row[key] = summary.get(key)
        business = summary.get("schema") == "executorch.business.summary.v1"
        if business:
            row["normal_rate"] = summary.get("pass_rate")
        reviewed = bool(row["planned"]) and row["reviewed"] == row["planned"]
        row.update(
            metric="business_generation" if business else "generation",
            status="reviewed" if reviewed else "pending_review",
        )
        if not reviewed:
            row.update(normal_rate=None, identity_correct_rate=None)
    return row


def main():
    """Publish the complete comparison directory atomically."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "Choose a new report directory; existing reports are not overwritten"
        )
    rows = [collect(path.resolve()) for path in args.runs]
    lines = [
        "# 模型验证汇总",
        "",
        "只汇总显式指定的目录；空值表示未完成、待复核或无有效性证据，不表示零分。",
        "不同语料/tokenizer/上下文/精度的 PPL 不能直接排名；此表不自动断言运行间可比。",
        "",
        "| 运行 | 状态 | 指标 | PPL | 已尝试/计划 | 已复核 | 严格通过率 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:

        def cell(value):
            return (
                "—"
                if value is None
                else str(value).replace("|", "\\|").replace("\n", " ")
            )

        lines.append(
            "| "
            + " | ".join(
                cell(value)
                for value in (
                    row["run"],
                    row["status"],
                    row["metric"],
                    row["ppl"],
                    f"{cell(row['attempted'])}/{cell(row['planned'])}",
                    row["reviewed"],
                    row["normal_rate"],
                )
            )
            + " |"
        )
    with atomic_directory(args.output) as working:
        with (working / "comparison.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (working / "comparison.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (working / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
