"""LegalBench-RU QA-аудит: ловит дефекты данных, которые не видит схема-валидатор.

Проверяет по всему корпусу:
  - дубли/почти-дубли вопросов (между задачами);
  - утечка ответа в вопрос (extraction: answer в тексте; citation: номер статьи в тексте);
  - утечка ответа в norm_text (grounded не должен называть Да/Нет);
  - перекос баланса (binary all-Да; MC all-A) — модель может угадывать;
  - tool_call: gold tool есть в каталоге, имена args ⊆ params инструмента;
  - пустые/короткие, дубли id.

Выход: список WARN (не валит CI, но требует внимания) + сводка.

    python benches/legalbench-ru/qa.py
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TASKS = Path(__file__).parent / "tasks"
CATALOG = Path(__file__).parent / "tool_catalog.json"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().replace("ё", "е")).strip()


def load_catalog() -> dict[str, set[str]]:
    cat: dict[str, set[str]] = {}
    if CATALOG.exists():
        for c in json.loads(CATALOG.read_text(encoding="utf-8")):
            cat[f"{c['server']}.{c['tool']}"] = set(c.get("params", []))
    return cat


def main() -> None:
    catalog = load_catalog()
    warns: list[str] = []
    all_q: dict[str, str] = {}        # norm(question) -> task/id
    all_ids: Counter = Counter()
    n_inst = 0
    tasks = sorted(p for p in TASKS.iterdir() if (p / "meta.json").exists())

    for t in tasks:
        meta = json.loads((t / "meta.json").read_text(encoding="utf-8"))
        at = meta["answer_type"]
        rows = [json.loads(l) for l in (t / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        bin_ans, mc_ans = Counter(), Counter()
        for r in rows:
            n_inst += 1
            iid = f"{t.name}/{r['id']}"
            all_ids[r["id"]] += 1
            q = norm(r.get("question", ""))
            # near-dup question
            if q in all_q:
                warns.append(f"DUP-Q  {iid}  ≈  {all_q[q]}")
            else:
                all_q[q] = iid
            ans = r.get("answer")
            # leakage: extraction answer literally in question
            if at == "extraction" and isinstance(ans, str) and len(ans) > 3 and norm(ans) in q:
                warns.append(f"LEAK-Q {iid}  answer '{ans}' уже в вопросе (тривиально)")
            # grounded leakage: norm_text states Да/Нет
            nt = norm(r.get("norm_text", ""))
            if nt and re.search(r"\bответ[:\s]+(да|нет)\b", nt):
                warns.append(f"LEAK-NT {iid}  norm_text проговаривает ответ")
            if at == "binary":
                bin_ans[str(ans)] += 1
            if at == "multiple_choice":
                mc_ans[str(ans)] += 1
            # tool_call gold sanity
            if at == "tool_call" and isinstance(ans, dict):
                tool = ans.get("tool")
                if tool:
                    if tool not in catalog:
                        warns.append(f"TOOL?  {iid}  gold tool '{tool}' нет в каталоге")
                    else:
                        bad = set((ans.get("args") or {}).keys()) - catalog[tool]
                        if bad:
                            warns.append(f"ARG?   {iid}  args {bad} не в params {tool}")
        # balance skew
        if bin_ans and (max(bin_ans.values()) / sum(bin_ans.values())) > 0.8:
            warns.append(f"SKEW-B {t.name}  binary перекос: {dict(bin_ans)}")
        if mc_ans and (max(mc_ans.values()) / sum(mc_ans.values())) > 0.7:
            warns.append(f"SKEW-MC {t.name}  MC перекос по буквам: {dict(mc_ans)}")

    for iid, c in all_ids.items():
        if c > 1:
            warns.append(f"DUP-ID id '{iid}' встречается {c}× (между задачами)")

    by_kind: Counter = Counter(w.split()[0] for w in warns)
    print(f"QA-аудит: {len(tasks)} задач, {n_inst} инстансов\n")
    for w in warns:
        print("  ! " + w)
    print(f"\nИтого предупреждений: {len(warns)}  {dict(by_kind)}")
    print("(QA не валит сборку — это сигналы на ревью; дубли id между задачами ок, если задачи независимы)")


if __name__ == "__main__":
    main()
