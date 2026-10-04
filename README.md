# Bebas Bias analyzer

The analysis engine behind **Biaskop**, which scores Indonesian news articles
for political and structural slant. This is the code that produced the verdicts
on the site. It's published so anyone can read how a score is made, then check
any published score against it.

> **This repository is generated.** Every time the engine is deployed, the
> files under `app/`, `scripts/` and `tests/` are copied byte for byte from the
> private backend repository. [`UPSTREAM.json`](UPSTREAM.json) names the
> backend commit they came from. Changes made here directly are overwritten by
> the next sync; see [CONTRIBUTING.md](CONTRIBUTING.md) for how to propose one.

## How a verdict is made

```text
headline + body ──► NLP signals ───┐
 (app/records.py)   (app/nlp.py)   │
                                   ├──► prompt ──────► model ──────► verdict
publisher domain ─► ownership ─────┘   (app/prompt.py) (app/llm.py)  (app/schemas.py)
                    context (app/publishers.py)
```

1. **Input.** The headline and body text the scraper extracted, the
   publisher's domain, and the URL. Production sends at most the first 6,000
   characters of the body: in Indonesian news, framing sits at the front.
2. **Local signals** (`app/nlp.py`). A *clickbait delta* (how much more
   emotionally intense the headline is than the body) and a list of charged
   words found in the text. They're lexicon-based, lightweight, and given to
   the model only as hints.
3. **Publisher context** (`app/publishers.py`). The publisher's owner, parent
   conglomerate and known political alignment, from the site's ownership graph.
   The model is told to use it to understand a lean, but to score only on the
   text.
4. **The rubric** (`app/prompt.py`). The system prompt *is* the methodology:
   the five axes, what 0, 50 and 100 mean on each, and the calibration rules.
   Every verdict is stamped with the `RUBRIC_VERSION` that produced it.
5. **One model call per article** (`app/llm.py`). OpenAI Chat Completions in
   JSON mode. Production runs `gpt-6-luna` with `reasoning_effort=low`.
   Large backlogs go through OpenAI's Batch API instead, with an identical
   request body.
6. **Parsing.** Scores must be integers from 0 to 100. Evidence snippets that
   name an unknown axis are dropped.

### The five axes

Each axis runs from 0 to 100. The neutral default is 50, except
sensationalism, where it's 10. The prompt is the authoritative definition; this
table is a summary.

| Axis | 0 | 100 |
|---|---|---|
| `elite_alignment` (patronage) | Critical of the ruling coalition; foregrounds dissent and corruption probes | Defensive of the ruling coalition, including by omission: filler that crowds out protests and accountability coverage |
| `identity_lens` (aliran) | Demagogic secular-nationalist framing | Demagogic religious-conservative framing. Informative coverage of national consensus (e.g. Palestine solidarity) stays at 50 |
| `economic_sovereignty` (market) | Open market, foreign investment | Resource nationalism, protectionism, downstreaming |
| `state_vs_civil` (authority) | Civil liberties, rights monitors given space | State security, including "verbatim police laundering": press releases as copy, suspects named pre-trial, no defense voice |
| `sensationalism` (commercial) | Dry, analytical | Deliberately misleading headline plus outrage bait. A straightforward headline caps the score at 35 |

`reliability_index` (0–100) measures sourcing: named and verified sources,
factual specificity, no unverified speculation.

## Setup

Python 3.12 or newer (production runs 3.14).

```bash
git clone https://github.com/seanmic1/bebas-bias-analyzer.git
cd bebas-bias-analyzer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/bootstrap_nltk.py   # NLTK data for the NLP signals, into ~/nltk_data
pytest
```

## Backtest a published verdict

Every article page on the site has the article's id in its link
(`.../article/<id>`). Either the id or the whole link works below, and so does
the publisher's URL for the article.

**See a verdict and exactly what produced it.** This is free and needs no API key:

```bash
python -m scripts.backtest show <article>
python -m scripts.backtest show <article> --request   # plus the exact model request
```

`show` prints the published scores, the model and rubric revision that
produced them, the publisher context, and the NLP signals. The signals are
recomputed from the text and compared with the ones stored alongside the verdict.
A match confirms that the text and the NLP code are the same ones production
ran. `--request` prints the request body the engine builds for that article, so
you can send it to the model yourself.

**Re-score it.** This calls the model, so it needs an OpenAI API key
(`cp .env.example .env`, then set `OPENAI_API_KEY`):

```bash
python -m scripts.backtest run <article> --repeats 3
python -m scripts.backtest run --latest 20 --publisher detik.com
```

Each article is re-scored `--repeats` times, by default with the model that
produced its published verdict. `--latest N` takes the most recently analyzed
articles. By default it keeps only verdicts made under this engine's rubric, so
they should reproduce. With `--rubric any`, the comparison measures how a
rubric change moved the scores instead. Every re-scoring is one model call
billed to your key: `--latest 20 --repeats 3` is 60 calls.

A report lands in `outputs/backtest_<time>.md` (and `.json`). For example:

| metric | published | re-scorings | mean | delta | noise | flag |
|---|---:|---|---:|---:|---:|:--:|
| `elite_alignment` | 75 | 72, 75 | 73.5 | -1.5 | 3.0 |  |
| `state_vs_civil` | 85 | 78, 82 | 80.0 | -5.0 | 4.0 |  |
| `sensationalism` | 30 | 25, 40 | 32.5 | +2.5 | 15.0 |  |
| `reliability_index` | 62 | 62, 58 | 60.0 | -2.0 | 4.0 |  |

The model doesn't give the same answer twice, so a published score is one draw
from a distribution. `noise` is the scorer's own spread across re-scorings of
identical input. A score is **flagged** when the re-scorings move away from it
by more than 1.5 × that noise *and* by more than 5 points. Averaged over many
articles, `mean delta` shows systematic drift between the published verdicts
and the engine as it is now.

What a backtest can't control for:

- **Publisher context isn't stored with the verdict.** The backtest uses
  today's ownership graph. If the graph changed since, the input changed too.
- **The text is what the scraper extracted**, as stored on the site, not the
  live page. Extraction itself happens in the backend and isn't checked here.
- **Reasoning effort isn't stored per verdict.** The engine's default (`low`)
  is what production uses.
- **A retired model can't be re-run.** Use `--model` to compare against a
  current one, and read the result as a comparison, not a reproduction.

## Audit the scorer for demographic bias

The counterfactual audit asks whether the demographic group an article names
moves its scores. It swaps the group for a counterpart from the same category
(one ethnicity for another, one party for another), re-scores both versions
several times, and compares the difference with the scorer's own noise. See
`app/audit/__init__.py` for the method and its limits.

```bash
python -m scripts.counterfactual_audit plan                # free: which pairs the articles allow
python -m scripts.counterfactual_audit run --limit 20      # calls the model
```

Articles come from the site's 200 most recently published ones
(`--articles N` for more, up to 1000) or from a file (`--from-json`). The
group lists live in `app/audit/data/groups.id.json`.

## Data access

The tools read the site's public data through Supabase's REST API, using the
publishable key the site already ships to every browser
(`app/public_data.py`). Row-level security governs what that key can do; the
tools only read articles, their verdicts and the ownership graph. To point the
tools at a fork's own
Supabase project, set `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY`.

This repository contains no article text. Articles belong to their publishers;
the tools fetch an article's stored text only to analyze it.

## What isn't here

The private backend also holds the scraper (feed discovery, page fetching,
text extraction, story clustering), the scheduled jobs that write verdicts to
the database, and the deployment. None of them take part in scoring: they
decide which articles get analyzed, and they store the result.

## License

[GNU AGPL-3.0](LICENSE), the same license as the backend.
