"""LegalBench-RU: валидация задач (схема + контент-линт). Без зависимостей.

Гоняется в CI. Exit code != 0 при ошибках. Предупреждения (WARN) не валят сборку,
но видны (например, инстанс без provenance gold_norm).

    python benches/legalbench-ru/validate.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TASKS_DIR = Path(__file__).parent / "tasks"

REASONING_TYPES = {"rule-recall", "rule-application", "rule-conclusion",
                   "issue-spotting", "interpretation", "rhetorical", "citation",
                   "tool-use", "comprehension"}
ANSWER_TYPES = {"extraction", "binary", "multiple_choice", "norm_citation", "open", "tool_call"}
# какой скоринг допустим для какого типа ответа
SCORING_FOR = {
    "extraction": "normalized_match",
    "binary": "binary_accuracy",
    "multiple_choice": "choice_accuracy",
    "norm_citation": "norm_f1",
    "open": "term_coverage",
    "tool_call": "tool_match",
}
ID_RE = re.compile(r"^[a-z0-9-]+$")
VER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
META_REQUIRED = ["id", "title", "reasoning_type", "domain", "answer_type", "scoring",
                 "needs_expert_review", "version"]

errors: list[str] = []
warns: list[str] = []


def err(task: str, msg: str) -> None:
    errors.append(f"  ✗ [{task}] {msg}")


def warn(task: str, msg: str) -> None:
    warns.append(f"  ! [{task}] {msg}")


def validate_meta(folder: str, meta: dict) -> str | None:
    for field in META_REQUIRED:
        if field not in meta:
            err(folder, f"meta.json: нет обязательного поля '{field}'")
    rt, at, scoring = meta.get("reasoning_type"), meta.get("answer_type"), meta.get("scoring")
    if meta.get("id") != folder:
        err(folder, f"meta.id '{meta.get('id')}' != имя папки '{folder}'")
    if rt and rt not in REASONING_TYPES:
        err(folder, f"неизвестный reasoning_type '{rt}'")
    if at and at not in ANSWER_TYPES:
        err(folder, f"неизвестный answer_type '{at}'")
    if at and scoring and SCORING_FOR.get(at) != scoring:
        err(folder, f"scoring '{scoring}' не соответствует answer_type '{at}' "
                    f"(ожидался '{SCORING_FOR.get(at)}')")
    if meta.get("version") and not VER_RE.match(meta["version"]):
        err(folder, f"version '{meta['version']}' не semver")
    if meta.get("needs_expert_review") is False and not meta.get("reviewed_by"):
        warn(folder, "needs_expert_review=false, но нет reviewed_by — кто валидировал?")
    return at


def validate_instances(folder: str, at: str | None, path: Path) -> None:
    if not path.exists():
        err(folder, "нет test.jsonl")
        return
    seen: set[str] = set()
    n = 0
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        n += 1
        try:
            inst = json.loads(line)
        except json.JSONDecodeError as e:
            err(folder, f"test.jsonl:{ln} битый JSON: {e}")
            continue
        iid = inst.get("id", "")
        if not ID_RE.match(iid):
            err(folder, f"test.jsonl:{ln} id '{iid}' не kebab-case")
        if iid in seen:
            err(folder, f"test.jsonl:{ln} дубль id '{iid}'")
        seen.add(iid)
        if not inst.get("question"):
            err(folder, f"[{iid}] нет question")
        if "answer" not in inst:
            err(folder, f"[{iid}] нет answer")
            continue
        ans = inst["answer"]
        if at == "multiple_choice":
            choices = inst.get("choices")
            if not isinstance(choices, list) or len(choices) < 2:
                err(folder, f"[{iid}] multiple_choice без choices")
            elif ans not in list("ABCD"[: len(choices)]):
                err(folder, f"[{iid}] answer '{ans}' вне диапазона вариантов")
        elif at == "binary":
            if str(ans) not in {"Да", "Нет"}:
                err(folder, f"[{iid}] binary answer должен быть 'Да'/'Нет', а не '{ans}'")
        elif at == "norm_citation":
            if not isinstance(ans, list) or not ans:
                err(folder, f"[{iid}] norm_citation answer должен быть непустым списком")
        elif at == "tool_call":
            if not isinstance(ans, dict) or "tool" not in ans:
                err(folder, f"[{iid}] tool_call answer должен быть объектом с ключом 'tool'")
        # provenance / review lints
        if at in {"extraction", "binary", "norm_citation"} and not inst.get("gold_norm") and at != "norm_citation":
            warn(folder, f"[{iid}] нет gold_norm (provenance для ревьюера)")
    if n == 0:
        err(folder, "test.jsonl пустой")
    else:
        print(f"  ✓ {folder:34s} {n} инстансов")


def main() -> None:
    folders = sorted(p for p in TASKS_DIR.iterdir() if p.is_dir())
    if not folders:
        raise SystemExit("нет задач в tasks/")
    print(f"Валидация LegalBench-RU: {len(folders)} задач\n")
    for folder in folders:
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            err(folder.name, "нет meta.json")
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            err(folder.name, f"битый meta.json: {e}")
            continue
        at = validate_meta(folder.name, meta)
        validate_instances(folder.name, at, folder / "test.jsonl")

    if warns:
        print("\nПредупреждения:")
        print("\n".join(warns))
    if errors:
        print("\nОшибки:")
        print("\n".join(errors))
        print(f"\n✗ FAIL: {len(errors)} ошибок, {len(warns)} предупреждений")
        sys.exit(1)
    print(f"\n✓ OK: 0 ошибок, {len(warns)} предупреждений")


if __name__ == "__main__":
    main()
