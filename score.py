"""LegalBench-RU: раннер + скоринг.

Прогоняет инстансы задач через модель и считает метрики по типу ответа.

Примеры:
    python benches/legalbench-ru/score.py --model stub
    python benches/legalbench-ru/score.py --model oracle           # верх. граница: должно дать ~100%
    python benches/legalbench-ru/score.py --model openai --model-name t-pro --reasoning-type citation
    python benches/legalbench-ru/score.py --model stub --task rule-recall-civil-terms --json out.json

Метрики (все приведены к [0,1] на инстанс, усредняются по группам):
    extraction      → normalized substring match (1/0)
    binary          → accuracy (1/0) + balanced accuracy по классам
    multiple_choice → accuracy (1/0)
    norm_citation   → set-F1 по нормализованным ссылкам «кодекс+статья»
    open            → term coverage (доля встретившихся must_mention)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models import get_model  # noqa: E402

TASKS_DIR = Path(__file__).parent / "tasks"
_CATALOG_PATH = Path(__file__).parent / "tool_catalog.json"


def _tool_catalog() -> list[dict]:
    return json.loads(_CATALOG_PATH.read_text(encoding="utf-8")) if _CATALOG_PATH.exists() else []


def _catalog_text() -> str:
    return "\n".join(
        f"- {c['server']}.{c['tool']}({', '.join(c.get('params', []))}) — {c.get('desc', '')}"
        for c in _tool_catalog()
    )


def _extract_json_obj(text: str) -> dict | None:
    """Первый сбалансированный {...}, который парсится как JSON-объект."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:  # noqa: BLE001
                        pass
                    break
        start = text.find("{", start + 1)
    return None


# ---------------------------------------------------------------- нормализация
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().replace("ё", "е")).strip()


# код кодекса/закона → канон. ключи — подстроки, ищутся в нормализованном тексте.
# Порядок важен: более специфичные/длинные ключи раньше коротких (гпк перед гк).
_CODE_ALIASES: list[tuple[str, str]] = [
    ("зозпп", "ЗоЗПП"), ("защите прав потреб", "ЗоЗПП"), ("2300-1", "ЗоЗПП"),
    ("о банкротстве", "ФЗ-127"), ("о несостоятельност", "ФЗ-127"),
    ("конституц", "Конституция"),
    ("гпк", "ГПК"), ("апк", "АПК"), ("кас", "КАС"), ("коап", "КоАП"),
    ("упк", "УПК"),
    ("гражданск", "ГК"), ("трудов", "ТК"), ("налогов", "НК"),
    ("семейн", "СК"), ("жилищн", "ЖК"), ("земельн", "ЗК"), ("бюджетн", "БК"),
    ("гк", "ГК"), ("тк", "ТК"), ("нк", "НК"), ("ск", "СК"),
    ("жк", "ЖК"), ("зк", "ЗК"), ("бк", "БК"), ("ук", "УК"),
]
_ART_RE = re.compile(r"(?:ст|стать[ияеёю])\.?\s*№?\s*(\d+(?:\.\d+)?)")
# Федеральный закон: «фз-115», «115-фз», «фз № 152». Канон → «ФЗ-<номер>».
_FZ_RE = re.compile(r"фз\s*[-№\s]*(\d+)(?:-фз)?|(\d+)\s*-\s*фз")


def _code_positions(text: str) -> list[tuple[int, str]]:
    """Позиции всех упоминаний кодов (named + ФЗ-номера), отсортированы по offset."""
    out: list[tuple[int, str]] = []
    for sub, canon in _CODE_ALIASES:
        start = 0
        while (i := text.find(sub, start)) >= 0:
            out.append((i, canon))
            start = i + 1
    for m in _FZ_RE.finditer(text):
        num = m.group(1) or m.group(2)
        if num:
            out.append((m.start(), f"ФЗ-{num}"))
    out.sort()
    return out


def extract_norms(text: str) -> set[tuple[str, str]]:
    """Достаёт множество пар (кодекс, статья) из произвольного текста.

    Каждый номер статьи привязывается к БЛИЖАЙШЕМУ ПРЕДШЕСТВУЮЩЕМУ коду
    («ФЗ-208 ст.78» → ФЗ-208), а при отсутствии кода слева — к первому коду в
    тексте. Распознаёт именованные кодексы (ГК/ТК/НК/ГПК/…) и ФЗ вида ФЗ-N.
    """
    t = norm(text)
    positions = _code_positions(t)
    pairs: set[tuple[str, str]] = set()
    for m in _ART_RE.finditer(t):
        preceding = [c for p, c in positions if p <= m.start()]
        code = preceding[-1] if preceding else (positions[0][1] if positions else None)
        if code is not None:
            pairs.add((code, m.group(1)))
    return pairs


# ------------------------------------------------------------------- скореры
def _as_list(x) -> list[str]:
    return x if isinstance(x, list) else [x]


def score_extraction(inst: dict, output: str) -> float:
    o = norm(output)
    gold = [str(inst["answer"]), *inst.get("accept", [])]
    return 1.0 if any(norm(g) in o for g in gold) else 0.0


def _extract_yes_no(output: str) -> str | None:
    for tok in re.findall(r"\b(да|нет)\b", norm(output)):
        return tok
    return None


def score_binary(inst: dict, output: str) -> float:
    pred = _extract_yes_no(output)
    gold = norm(str(inst["answer"]))
    return 1.0 if pred == gold else 0.0


def _extract_choice(inst: dict, output: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", output)
    if m:
        return m.group(1)
    o = norm(output)  # fallback: матч по тексту варианта
    for i, choice in enumerate(inst.get("choices", [])):
        if norm(choice) and norm(choice) in o:
            return "ABCD"[i]
    return None


def score_choice(inst: dict, output: str) -> float:
    return 1.0 if _extract_choice(inst, output) == str(inst["answer"]) else 0.0


def score_norm_f1(inst: dict, output: str) -> float:
    gold = {p for g in _as_list(inst["answer"]) for p in extract_norms(g)}
    pred = extract_norms(output)
    if not gold:
        return 0.0
    inter = len(gold & pred)
    prec = inter / len(pred) if pred else 0.0
    rec = inter / len(gold)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def score_open(inst: dict, output: str) -> float:
    o = norm(output)
    terms = _as_list(inst["answer"])
    if not terms:
        return 0.0
    return sum(1 for t in terms if norm(t) in o) / len(terms)


def _norm_tool(t: str | None) -> str:
    return (t or "").strip().lower().split("(")[0]


def score_tool_call(inst: dict, output: str) -> float:
    """Tool-routing: 0 если не тот tool; 0.6 за верный tool + до 0.4 за аргументы.
    Для негативов (gold tool=null) — 1.0 если модель НЕ вызвала инструмент."""
    gold = inst["answer"] if isinstance(inst["answer"], dict) else {}
    gold_tool = gold.get("tool")
    obj = _extract_json_obj(output) or {}
    pred_tool = obj.get("tool")
    pred_args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
    if not gold_tool:  # негатив: правильно НЕ звать инструмент
        return 1.0 if (not pred_tool or str(pred_tool).lower() in {"none", "null"}) else 0.0
    g, p = _norm_tool(gold_tool), _norm_tool(pred_tool)
    if g != p and g.split(".")[-1] != p.split(".")[-1]:
        return 0.0
    gold_args = gold.get("args") or {}
    if not gold_args:
        return 1.0
    hit = sum(1 for k, v in gold_args.items()
              if k in pred_args and norm(str(pred_args[k])) == norm(str(v)))
    return 0.6 + 0.4 * (hit / len(gold_args))


SCORERS = {
    "extraction": score_extraction,
    "binary": score_binary,
    "multiple_choice": score_choice,
    "norm_citation": score_norm_f1,
    "open": score_open,
    "tool_call": score_tool_call,
}


# ---------------------------------------------------------------- prompt/oracle
def build_prompt(inst: dict, answer_type: str, mode: str = "closed") -> str:
    """Строит промпт под условие прогона.

    closed     — только вопрос (параметрическая память; knowledge-дорожка).
    grounded   — перед вопросом дословный текст применимой нормы (inst['norm_text']).
    distractor — вместо неё ПОДСТАВНАЯ норма (inst['distractor_text']) → counterfactual robustness.
    reject     — нормы нет + явное разрешение отказаться («Недостаточно данных») → negative rejection.
    Если для grounded/distractor нет нужного текста — деградирует до closed.
    tool_call — отдельный формат: каталог инструментов + запрос → JSON-вызов.
    """
    if answer_type == "tool_call":
        return (
            f"Доступные инструменты (MCP-серверы российских реестров и права):\n{_catalog_text()}\n\n"
            f"Запрос юриста: {inst['question']}\n\n"
            "Выбери ОДИН наиболее подходящий инструмент и его аргументы. "
            'Ответь СТРОГО одним JSON-объектом: {\"tool\": \"server.tool_name\", \"args\": {...}}. '
            'Если ни один инструмент не подходит — ответь {\"tool\": null}.'
        )
    parts: list[str] = []
    if mode == "grounded" and inst.get("norm_text"):
        parts.append(f"Применимая норма права (приведена дословно):\n«{inst['norm_text']}»\n")
    elif mode == "distractor" and inst.get("distractor_text"):
        parts.append(f"Норма права (приведена дословно):\n«{inst['distractor_text']}»\n")
    elif mode == "temporal" and inst.get("temporal_text"):
        # устаревшая редакция, поданная БЕЗ пометки — как если бы ретривер достал старый текст
        parts.append(f"Норма права (приведена дословно):\n«{inst['temporal_text']}»\n")
    elif mode == "reject":
        parts.append("Если предоставленных норм недостаточно для однозначного ответа, "
                     "ответь дословно: «Недостаточно данных».\n")
    if inst.get("context"):
        parts.append(f"Контекст: {inst['context']}\n")
    body = "\n".join([*parts, inst["question"]])
    if answer_type == "multiple_choice":
        opts = "\n".join(f"{'ABCD'[i]}) {c}" for i, c in enumerate(inst.get("choices", [])))
        return f"{body}\n{opts}\nОтветь одной буквой варианта."
    if answer_type == "binary":
        return body  # текст вопроса уже требует «Да или Нет»
    if answer_type == "norm_citation":
        return f"{body}\nУкажи нормативный акт и номер статьи."
    return f"{body}\nОтветь кратко, только по существу."


def task_track(meta: dict) -> str:
    """knowledge — память (citation/rule-recall); reasoning — рассуждение при норме;
    tool-use — выбор инструмента; comprehension — чтение поданного текста."""
    if meta.get("track"):
        return meta["track"]
    rt = meta["reasoning_type"]
    if rt == "tool-use":
        return "tool-use"
    if rt == "comprehension":
        return "comprehension"
    return "knowledge" if rt in {"citation", "rule-recall"} else "reasoning"


def oracle_output(inst: dict, answer_type: str) -> str:
    """«Идеальный» ответ из gold — для проверки, что скоринг даёт ~100%."""
    if answer_type == "norm_citation":
        return "; ".join(_as_list(inst["answer"]))
    if answer_type == "open":
        return " ".join(_as_list(inst["answer"]))
    if answer_type == "tool_call":
        return json.dumps(inst["answer"], ensure_ascii=False)
    return str(inst["answer"])


# ---------------------------------------------------------------- загрузка задач
def load_tasks() -> list[dict]:
    tasks = []
    for meta_path in sorted(TASKS_DIR.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        test_path = meta_path.with_name("test.jsonl")
        instances = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        meta["_instances"] = instances
        tasks.append(meta)
    return tasks


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="LegalBench-RU runner")
    ap.add_argument("--model", default="stub", help="stub | oracle | openai")
    ap.add_argument("--model-name", default="", help="имя модели для --model openai")
    ap.add_argument("--mode", default="closed",
                    choices=["closed", "grounded", "distractor", "temporal", "reject"],
                    help="условие прогона: closed (память) | grounded (верная норма) | "
                         "distractor (подставная неверная норма) | temporal (устаревшая редакция) | "
                         "reject (нормы нет, можно отказаться)")
    ap.add_argument("--task", default="", help="id одной задачи")
    ap.add_argument("--reasoning-type", default="", help="фильтр по типу рассуждения")
    ap.add_argument("--track", default="", help="фильтр по дорожке: knowledge | reasoning")
    ap.add_argument("--domain", default="", help="фильтр по домену")
    ap.add_argument("--limit", type=int, default=0, help="ограничить число инстансов на задачу")
    ap.add_argument("--json", default="", help="сохранить детальный отчёт")
    args = ap.parse_args()

    model = None if args.model == "oracle" else get_model(args.model, args.model_name)
    model_name = "oracle" if args.model == "oracle" else model.name

    tasks = load_tasks()
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]
    if args.reasoning_type:
        tasks = [t for t in tasks if t["reasoning_type"] == args.reasoning_type]
    if args.track:
        tasks = [t for t in tasks if task_track(t) == args.track]
    if args.domain:
        tasks = [t for t in tasks if t["domain"] == args.domain]
    if not tasks:
        raise SystemExit("нет задач под фильтр")

    print(f"LegalBench-RU · модель={model_name} · режим={args.mode} · задач={len(tasks)}\n")
    results = []
    needs_review = 0
    grounded_missing = 0
    n_errors = 0
    for task in tasks:
        at = task["answer_type"]
        track = task_track(task)
        scorer = SCORERS[at]
        insts = task["_instances"][: args.limit] if args.limit else task["_instances"]
        task_scores = []
        bin_pred, bin_gold = [], []
        for inst in insts:
            if args.mode == "grounded" and not inst.get("norm_text"):
                grounded_missing += 1
            prompt = build_prompt(inst, at, args.mode)
            try:
                output = oracle_output(inst, at) if model is None else model.generate(prompt)
            except Exception as e:  # noqa: BLE001 — один сбойный вызов не должен валить прогон
                n_errors += 1
                results.append({
                    "task": task["id"], "reasoning_type": task["reasoning_type"],
                    "track": track, "domain": task["domain"], "instance": inst["id"],
                    "score": None, "error": str(e)[:160], "output": "",
                })
                continue
            sc = scorer(inst, output)
            task_scores.append(sc)
            if inst.get("needs_expert_review") or task.get("needs_expert_review"):
                needs_review += 1
            if at == "binary":
                bin_pred.append(_extract_yes_no(output))
                bin_gold.append(norm(str(inst["answer"])))
            results.append({
                "task": task["id"], "reasoning_type": task["reasoning_type"],
                "track": track, "domain": task["domain"], "instance": inst["id"],
                "score": round(sc, 3), "output": output[:200],
            })
        mean = sum(task_scores) / len(task_scores) if task_scores else 0.0
        extra = ""
        if at == "binary":
            extra = f"  bal_acc={_balanced_accuracy(bin_pred, bin_gold):.2f}"
        print(f"  {task['id']:34s} [{at:15s}] {mean:.2f}  (n={len(task_scores)}){extra}")

    scored = [r for r in results if r["score"] is not None]
    overall = sum(r["score"] for r in scored) / len(scored) if scored else 0.0
    print("\n" + "=" * 64)
    print(f"  OVERALL score : {overall:.2%}  (инстансов: {len(scored)}, режим={args.mode})")
    print("  по дорожкам:")
    for tr in sorted({r["track"] for r in scored}):
        vals = [r["score"] for r in scored if r["track"] == tr]
        print(f"    {tr:16s} {sum(vals)/len(vals):.2%}  (n={len(vals)})")
    print("  по типам рассуждения:")
    for rt in sorted({r["reasoning_type"] for r in scored}):
        vals = [r["score"] for r in scored if r["reasoning_type"] == rt]
        print(f"    {rt:16s} {sum(vals)/len(vals):.2%}  (n={len(vals)})")
    if n_errors:
        print(f"\n  ⚠ ошибок API (исключены из метрики): {n_errors}")
    if args.mode == "grounded" and grounded_missing:
        print(f"\n  ⚠ grounded: у {grounded_missing}/{len(results)} инстансов нет norm_text "
              f"— прогнаны как closed (норму нужно дозаполнить)")
    print(f"\n  ⚠ инстансов с needs_expert_review: {needs_review}/{len(results)} "
          f"— цифры предварительные до юр-валидации")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "model": model_name, "overall": overall, "n": len(results), "results": results,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nдетальный отчёт → {args.json}")


def _balanced_accuracy(pred: list[str | None], gold: list[str]) -> float:
    classes = set(gold)
    if not classes:
        return 0.0
    recalls = []
    for c in classes:
        idx = [i for i, g in enumerate(gold) if g == c]
        if idx:
            recalls.append(sum(1 for i in idx if pred[i] == c) / len(idx))
    return sum(recalls) / len(recalls) if recalls else 0.0


if __name__ == "__main__":
    main()
