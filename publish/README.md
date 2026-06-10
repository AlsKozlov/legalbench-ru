---
license: cc-by-4.0
language:
- ru
task_categories:
- question-answering
- text-classification
tags:
- legal
- russian-law
- legal-reasoning
- benchmark
- tool-use
- llm-evaluation
size_categories:
- n<1K
pretty_name: LegalBench-RU
configs:
- config_name: default
  data_files: legalbench_ru.jsonl
---

# LegalBench-RU

**Первый открытый многотрековый бенчмарк правового рассуждения для российского права.**

846 задач, 4 трека: знание права (knowledge), рассуждение (reasoning), выбор
инструмента к госреестрам (tool-use), устойчивость к устаревшим нормам (temporal).
Собран многоагентной генерацией с заземлением в реальных источниках +
адверсариальной верификацией; измеренный red-team error-rate — **3%**.

> ⚠️ **v0.3 / release candidate.** Все кейсы помечены `needs_expert_review`.
> Бенч AI-сконструирован и адверсариально верифицирован, но **юр-ревью живым
> юристом ещё не пройдено** → baseline-цифры предварительные. Прозрачность — часть
> дизайна (см. отчёты аудита в репозитории).

## Что внутри

| трек | кейсов | что меряет | источник |
|---|---|---|---|
| **knowledge** | 311 | память: статья/норма (citation, rule-recall) | lawyer-authored skills |
| **reasoning** | 348 | вывод по фактам, толкование, расчёты, позиции судов | skills + 45 ПП ВС + 27 ПКС + детерм. арифметика |
| **tool-use** | 187 | query → правильный MCP-вызов к госреестру | каталог 79 инструментов 17 серверов |
| **temporal** | (в reasoning) | устойчивость к устаревшей редакции | курировано |

16 правовых доменов: гражданское, договорное, трудовое, налоговое, корпоративное,
IP, банкротство/litigation, защита данных, потребительское, недвижимость,
закупки, AML/комплаенс, AI-governance, административное и др.

## Формат

Один JSON-объект на строку (`legalbench_ru.jsonl`). Поля:
- `task`, `domain`, `reasoning_type`, `answer_type`, `track`, `split`;
- `question`, `context?`, `choices?` (для multiple_choice);
- `answer` — gold (строка / буква / список норм / `{tool,args}`);
- `accept?`, `gold_norm?` (provenance), `explanation?`;
- `norm_text?` / `distractor_text?` / `temporal_text?` — для условий grounded /
  distractor / temporal;
- `needs_expert_review`.

### Сплиты и анти-контаминация
- `split=public` (674) — открытый.
- `split=holdout` (172, ~20%) — рекомендуется держать **ответы приватно** для
  чистых замеров; в пакете есть `holdout_questions_only.jsonl` (без gold) для
  сабмишена предсказаний.
- В начало `legalbench_ru.jsonl` встроена **canary-строка** (`_canary`) — чтобы
  потом детектировать утечку датасета в обучающих корпусах. **Не включайте этот
  датасет в обучение.**

## Условия прогона (главная фишка)

Один и тот же reasoning-кейс гоняется при разном поданном контексте:
`closed` (память) / `grounded` (верная норма) / `distractor` (подставная неверная)
/ `temporal` (устаревшая редакция) / `reject` (нормы нет). Ценность — в дельтах:
`closed→grounded` (упор в память ⇒ нужен RAG), `grounded→distractor` (ведётся ли
на ложную норму), `temporal` (анкорится ли на старый закон).

## Скоринг

Детерминированный (без LLM-судьи на скоринге → воспроизводимо): exact-match
(binary/MC), normalized-match (extraction), **norm-F1** (citation), **AST-match**
(tool-call). Харнесс — в репозитории; oracle-прогон = 100%, stub = 0%.

```bash
python score.py --model oracle                       # проверка скорера
python score.py --model openai --model-name <модель> # своя модель (OpenAI-совместимый endpoint)
python score.py --model openai --model-name <модель> --mode grounded --track reasoning
```

## Baseline (closed, весь бенч, предварительно)

| модель | overall | knowledge | reasoning | tool-use |
|---|---|---|---|---|
| DeepSeek-v32 | 76% | 53% | 91% | 85% |
| YandexGPT 5.1 | 76% | 62% | 85% | 82% |
| YandexGPT-5 Pro | 72% | 52% | 83% | 85% |
| Qwen3-235B | 71% | 48% | 85% | 84% |
| Qwen3.6 | 65% | 42% | 76% | 81% |
| GPT-OSS-120B | 57% | 21% | 76% | 80% |

Узкое место у всех — **знание** права (21–62%), не рассуждение (76–91%).

## Лицензия

Датасет — **CC-BY-4.0**. Код харнесса — Apache-2.0 (как репозиторий). Вопросы и
gold-ответы оригинальные; нормы РФ — общественное достояние.

## Цитирование

```bibtex
@misc{legalbench-ru-2026,
  title  = {LegalBench-RU: An Open Multi-Track Benchmark for Russian Legal Reasoning},
  author = {ru-legal contributors},
  year   = {2026},
  note   = {v0.3, https://github.com/AlsKozlov/legalbench-ru}
}
```
