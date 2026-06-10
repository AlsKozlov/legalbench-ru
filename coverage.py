"""Отчёт о покрытии LegalBench-RU: матрица домен × тип рассуждения + сводка.

    python benches/legalbench-ru/coverage.py
    python benches/legalbench-ru/coverage.py --md   # markdown-таблица для README
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TASKS_DIR = Path(__file__).parent / "tasks"
RT_ORDER = ["rule-recall", "rule-application", "rule-conclusion", "issue-spotting",
            "interpretation", "rhetorical", "citation", "tool-use", "comprehension"]


def load() -> list[dict]:
    out = []
    for meta_path in sorted(TASKS_DIR.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        test = meta_path.with_name("test.jsonl")
        n = sum(1 for line in test.read_text(encoding="utf-8").splitlines() if line.strip()) if test.exists() else 0
        meta["_n"] = n
        out.append(meta)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="markdown-таблица")
    args = ap.parse_args()

    tasks = load()
    grid: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    domains: set[str] = set()
    reviewed = 0
    total = 0
    for t in tasks:
        grid[t["domain"]][t["reasoning_type"]] += t["_n"]
        domains.add(t["domain"])
        total += t["_n"]
        if not t.get("needs_expert_review", True):
            reviewed += t["_n"]
    doms = sorted(domains)
    rts = [r for r in RT_ORDER if any(grid[d].get(r) for d in doms)]

    def cell(d, r):
        v = grid[d].get(r, 0)
        return str(v) if v else "·"

    if args.md:
        head = "| домен \\ тип | " + " | ".join(rts) + " | Σ |"
        sep = "|" + "---|" * (len(rts) + 2)
        print(head)
        print(sep)
        for d in doms:
            row = sum(grid[d].values())
            print(f"| {d} | " + " | ".join(cell(d, r) for r in rts) + f" | **{row}** |")
        tot_row = "| **Σ** | " + " | ".join(str(sum(grid[d].get(r, 0) for d in doms)) for r in rts) + f" | **{total}** |"
        print(tot_row)
    else:
        w = max(len(d) for d in doms) + 1
        print(f"LegalBench-RU coverage · задач={len(tasks)} · инстансов={total}\n")
        hdr = "domain".ljust(w) + "".join(r[:9].rjust(11) for r in rts) + "      Σ"
        print(hdr)
        print("-" * len(hdr))
        for d in doms:
            row = sum(grid[d].values())
            print(d.ljust(w) + "".join(cell(d, r).rjust(11) for r in rts) + f"{row:7d}")
        print("-" * len(hdr))
        print("Σ".ljust(w) + "".join(str(sum(grid[d].get(r, 0) for d in doms)).rjust(11) for r in rts) + f"{total:7d}")
        print(f"\nдоменов: {len(doms)} · типов рассуждения: {len(rts)}")
        print(f"валидировано юристом: {reviewed}/{total} "
              f"(остальное — needs_expert_review, цифры предварительные)")


if __name__ == "__main__":
    main()
