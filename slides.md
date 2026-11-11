---
marp: true
theme: default
paginate: true
size: 16:9
title: Prompt Registry & Local LLM Inference with MLflow 3.13
description: Versioning, aliasing and evaluating prompts with MLflow 3.13, using Llama 3.2 served by Ollama on Google Colab Pro
style: |
  :root {
    --ink: #222222;
    --muted: #666666;
    --untar: #8a0f1f;       /* UNTAR maroon */
    --untar-dark: #262626;  /* footer bar */
    --soft: #f4f4f4;
    --ok: #1a7f37;
  }
  section {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: var(--ink);
    font-size: 25px;
    /* bottom padding keeps content above the UNTAR footer bar */
    padding: 40px 64px 92px 64px;
    background-color: #ffffff;
    background-image: url("assets/untar_template.png");
    background-size: 100% 100%;
    background-repeat: no-repeat;
    place-content: start start !important;   /* top-align (Marp default theme centres vertically) */
  }
  h1 { color: var(--untar); font-size: 1.7em; margin: 0 0 0.3em; }
  h2 { color: var(--untar); font-size: 1.3em; margin: 0 0 0.45em; padding-bottom: 0.12em; border-bottom: 3px solid var(--untar); }
  h3 { color: var(--untar-dark); font-size: 0.95em; margin: 0.5em 0 0.2em; }
  strong { color: var(--untar); }
  em { color: var(--muted); }
  ul, ol { margin-top: 0.2em; }
  li { margin: 0.15em 0; }
  code { background: var(--soft); color: var(--untar); border-radius: 4px; padding: 0 0.25em; font-size: 0.9em; }
  pre { background: #1e1e1e; border-radius: 8px; font-size: 0.56em; line-height: 1.38; padding: 0.7em 0.9em; }
  pre code { background: transparent; color: #f0f0f0; padding: 0; }
  pre .hljs-keyword, pre .hljs-built_in { color: #ff7b72; }
  pre .hljs-string { color: #a5d6ff; }
  pre .hljs-number, pre .hljs-literal, pre .hljs-attr { color: #79c0ff; }
  pre .hljs-title, pre .hljs-function { color: #d2a8ff; }
  pre .hljs-comment { color: #8b949e; font-style: italic; }
  table { font-size: 0.7em; border-collapse: collapse; display: table !important; width: 100% !important; }
  th { background: var(--untar); color: #fff; }
  td, th { padding: 5px 10px; }
  tr:nth-child(even) td { background: var(--soft); }
  blockquote { border-left: 5px solid var(--untar); background: var(--soft); padding: 0.3em 0.9em; font-size: 0.82em; margin: 0.6em 0; }
  section::after { color: var(--muted); font-size: 0.55em; }
  /* page number: sits on the dark footer bar, clear of the maroon logo block */
  section::after { right: 345px; bottom: 12px; padding: 0; color: #d0d0d0; font-size: 0.6em; }

  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; align-items: start; }
  .small { font-size: 0.8em; }
  .ok { color: var(--ok); font-weight: 700; }
  .bad { color: var(--untar); font-weight: 700; }
  .center { text-align: center; }

  section.title { place-content: center start !important; padding-top: 20px; }
  section.title h1 { font-size: 1.9em; line-height: 1.2; }
  section.title h2 { border: none; color: var(--untar-dark); font-weight: 400; font-size: 1.1em; }
  section.dense { font-size: 21px; }
  section.dense pre { font-size: 0.6em; }
---

<!-- _class: title -->
<!-- _paginate: false -->

# Prompt Registry & Local LLM Inference with MLflow 3.13
## Versioning, aliasing and evaluating prompts with Llama 3.2 on Google Colab Pro

---

## Agenda

<div class="cols">
<div>

### Part 1 — Prompt Registry
- Why prompts need governance
- Register, load, version
- Aliases: promote and roll back
- Compare and evaluate versions

</div>
<div>

### Part 2 — Our implementation
- Why we run the model locally
- Ollama + Llama 3.2 on a Colab T4
- Typed output with Pydantic AI
- Results, limitations, lessons

</div>
</div>

> Base material: the MLflow 3.13 *Prompt Registry & AI Gateway* workshop. **Our change:** the OpenAI/Claude calls are replaced with a local open-weights model.

---

<!-- _class: title -->
<!-- _paginate: false -->

# Part 1
## Prompt Registry

Treat prompts like code and models: versioned, reviewable, reversible.

---

## Before: the prompt lives next to the code

```python
SYSTEM_PROMPT = """
Classify this support ticket into one of:
PAYMENT, REFUND, RESCHEDULE, or OTHER.
"""

def triage(ticket: str) -> str:
    response = client.chat.completions.create(
        model="llama3.2:3b",
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": ticket}],
    )
    return response.choices[0].message.content
```

Fine for a prototype. **It breaks down in a real project.**

---

<!-- _class: dense -->

## The problems

| Problem | What it looks like in practice |
| --- | --- |
| **No version history** | You "improve" a prompt, quality drops, and you cannot recover the old wording. |
| **Deployment coupling** | A one-word tweak needs a code change, PR, build and redeploy. |
| **No safe rollback** | Reverting means reverting a commit, not flipping a pointer. |
| **Hard to compare** | "What changed between the version that worked and the one that didn't?" |
| **No lineage** | Which prompt version produced this output or evaluation run? |
| **No quality loop** | Changes are judged by gut feeling, not by metrics on a dataset. |

> **Root cause:** prompts are treated as *incidental strings* rather than *versioned artifacts*.

---

## What the MLflow Prompt Registry offers

- **Version control** — every version is **immutable**, with a commit message
- **Aliases** — mutable pointers such as `production` to promote and roll back **without code changes**
- **Templates** — `{{ variables }}` keep instructions separate from input data
- **Lineage & evaluation** — plugs into `mlflow.genai.evaluate()` and MLflow tracking
- **Central UI** — browse, diff and manage prompts in one place

All APIs live under **`mlflow.genai`**.

---

## Core concepts

| Concept | Meaning |
| --- | --- |
| **Prompt** | A named, versioned template |
| **Template** | Text with `{{ variables }}` (Jinja2-style) |
| **Version** | Sequential and immutable (`v1`, `v2`, ...) |
| **Alias** | Mutable pointer, e.g. `@production` |
| **URI** | `prompts:/name/2` (exact) or `prompts:/name@production` (alias) |

```python
mlflow.genai.register_prompt(name=..., template=..., commit_message=...)
mlflow.genai.load_prompt("prompts:/workshop-ticket-triage@production")
mlflow.genai.set_prompt_alias(name, alias="production", version=2)
```

---

<!-- _class: title -->
<!-- _paginate: false -->

# Part 2
## Our implementation

Same governance workflow, but the model runs on our own GPU.

---

## Why we changed the model backend

<div class="cols">
<div>

### What we hit with cloud APIs
- No OpenAI / Claude key available
- Gemini free tier returned **HTTP 429**: limit of **5 requests / minute** per model
- Evaluation runs many calls in a burst, so quota is exhausted quickly
- `mlflow gateway start` is **not supported on Windows**

</div>
<div>

### What we did instead
- Run **Llama 3.2 (3B)** with **Ollama** on a **Colab Pro T4 GPU**
- Ollama exposes an **OpenAI-compatible** API at `localhost:11434/v1`
- Same `openai` SDK code, only `base_url` and `model` change
- No API key, no rate limit, data stays in the session

</div>
</div>

---

## Architecture

```text
 Google Colab Pro  (Tesla T4, 15 GB VRAM)
 ┌─────────────────────────────────────────────────────────────┐
 │  MLflow Tracking Server  (sqlite:///mlflow.db, port 5000)   │
 │    • Prompt Registry: v1, v2, v3  +  @production alias      │
 │    • Evaluation runs and metrics                            │
 └───────────────────────────▲─────────────────────────────────┘
                             │ load_prompt("...@production")
 ┌───────────────────────────┴─────────────────────────────────┐
 │  Notebook / application code                                │
 │    • openai SDK  and  Pydantic AI agent                     │
 └───────────────────────────┬─────────────────────────────────┘
                             │ OpenAI-compatible request
 ┌───────────────────────────▼─────────────────────────────────┐
 │  Ollama  →  llama3.2:3b   (http://localhost:11434/v1)       │
 └─────────────────────────────────────────────────────────────┘
```

---

<!-- _class: dense -->

## Setup: Ollama on Colab

```bash
apt-get install -y -qq pciutils zstd          # zstd is needed to unpack Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                                # start the daemon
ollama pull llama3.2:3b                       # download the model
```

```python
from openai import OpenAI

local_llm_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",            # placeholder; the SDK requires a value
)
```

> **Gotcha we hit:** the first install failed with `requires zstd` until it was installed.
> **GPU choice:** a standard **T4** is enough for a 3B model.

---

<!-- _class: dense -->

## Step 1 — Register version 1

```python
V1_TEMPLATE = """Classify this support ticket into one category:
PAYMENT, REFUND, RESCHEDULE, or OTHER.

Ticket: {{ ticket }}

Respond with the label only."""

prompt_v1 = mlflow.genai.register_prompt(
    name="workshop-ticket-triage",
    template=V1_TEMPLATE,
    commit_message="v1: Initial minimal classification prompt",
    tags={"task": "ticket-triage", "backend": "colab-gpu"},
)
# Registered workshop-ticket-triage version 1
# URI: prompts:/workshop-ticket-triage/1
```

The application fills the variable at call time: `loaded.format(ticket=...)`.

---

<!-- _class: dense -->

## Step 2 — Version 2 and a Git-style diff

<div class="cols">
<div>

```python
V2_TEMPLATE = """You are an expert support
ticket triage assistant.

Task: Classify the ticket into
exactly one of:
- PAYMENT: billing, duplicate charges
- REFUND: money back, cancellations
- RESCHEDULE: move appointment/date
- OTHER: general inquiries

Rules:
- Output exactly one uppercase label
- No explanation or punctuation

Ticket: {{ ticket }}"""
```

</div>
<div>

```diff
--- v1
+++ v2
-Classify this support ticket...
-PAYMENT, REFUND, RESCHEDULE...
+You are an expert support ticket
+triage assistant.
+Task: Classify the ticket into
+exactly one of:
+- PAYMENT: billing, ...
+- REFUND: money back, ...
+Rules:
+- Output exactly one uppercase label
 Ticket: {{ ticket }}
-Respond with the label only.
```

</div>
</div>

Produced with `difflib.unified_diff` on the two loaded templates (the UI has a Compare view too).

---

## Step 3 — Promote with an alias

<div class="cols">
<div>

```python
mlflow.genai.set_prompt_alias(
    "workshop-ticket-triage",
    alias="production",
    version=2,
)

prod = mlflow.genai.load_prompt(
    "prompts:/workshop-ticket-triage@production"
)
# alias @production -> version 2
```

</div>
<div>

### Why it matters
- App code references **`@production`**, never a version number
- Promote = move the pointer
- **Roll back** = move it back (`version=1`)
- No rebuild, no redeploy

> Alias lookups are cached briefly, so a change may take a moment to be seen by running apps.

</div>
</div>

---

<!-- _class: dense -->

## Step 4 — Governed prompt + local model

```python
def classify_ticket(ticket, prompt_uri=None):
    uri = prompt_uri or "prompts:/workshop-ticket-triage@production"
    prompt = mlflow.genai.load_prompt(uri)
    response = local_llm_client.chat.completions.create(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt.format(ticket=ticket)}],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()
```

| Ticket (output of our run) | Label |
| --- | --- |
| I was charged twice for one hotel night. | PAYMENT |
| Need to move my dental appointment to next Thursday. | RESCHEDULE |
| I want my money back for a cancelled flight. | REFUND |
| What are your customer support working hours? | OTHER |

---

<!-- _class: dense -->

## Step 5 — Typed output with Pydantic AI

```python
class TriageResult(BaseModel):
    category: Literal["PAYMENT", "REFUND", "RESCHEDULE", "OTHER"]
    reason: str

pydantic_model = OpenAIChatModel(
    "llama3.2:3b",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama"),
)

agent = Agent(pydantic_model, output_type=TriageResult,
              system_prompt="Classify customer support tickets accurately.")
result = agent.run_sync(prompt.format(ticket="I was charged twice for one hotel night."))
```

```text
Category : PAYMENT
Reason   : duplicate charges
```

The registry owns the **prompt**, the local server owns the **model**, Pydantic AI owns the **output schema**.

---

<!-- _class: dense -->

## Step 6 — Evaluate versions on a dataset

```python
EVAL_DATA = [{"inputs": {"ticket": "Charged twice for one hotel night."},
              "expectations": {"expected_response": "PAYMENT"}}, ...]   # 4 rows

@scorer
def category_match(outputs, expectations):          # contains the expected label?
    return float(expectations["expected_response"] in str(outputs).strip().upper())

@scorer
def label_guidelines(outputs):                      # is it exactly one valid label?
    return float(str(outputs).strip().upper().strip(".") in
                 {"PAYMENT", "REFUND", "RESCHEDULE", "OTHER"})

with mlflow.start_run(run_name="eval-v1-local"):
    mlflow.genai.evaluate(data=EVAL_DATA, predict_fn=make_predict_fn(uri_v1),
                          scorers=[category_match, label_guidelines])
```

Both scorers are **deterministic** (no LLM judge), so results are repeatable.

---

## Results

| Version | `category_match` | `label_guidelines` | Note |
| :---: | :---: | :---: | --- |
| **v1** | <span class="bad">0.50</span> | 1.00 | minimal prompt |
| **v2** | <span class="ok">1.00</span> | 1.00 | explicit labels and rules → promoted |
| **v3** | <span class="ok">1.00</span> | 1.00 | `@production` re-evaluated |

- Adding explicit label definitions and output rules raised `category_match` from **0.50 to 1.00** on our dataset.
- v3 matched v2; **the dataset cannot separate them**.
- v3 was written **by hand** as a refinement. We did not run `optimize_prompts()`.

---

## Limitations (what we should not over-claim)

- **Tiny dataset:** 4 examples, one per label. Good for a demo, not a statistical result.
- **Weak scorers:** `category_match` accepts the label anywhere in the text, so it is lenient.
- **Small model:** a 3B model is more sensitive to prompt wording; results may differ for larger models.
- **No AI Gateway in this run:** routing, fallbacks and usage tracking were covered conceptually; we called Ollama directly.
- **Session storage:** the MLflow SQLite DB lives in the Colab session and is lost on disconnect unless saved to Drive.

---

## Challenges and lessons learned

| Challenge | How we handled it |
| --- | --- |
| Gemini free tier: HTTP 429 (5 req/min) | Moved inference to a local model |
| `mlflow gateway start` fails on Windows | Ran everything on Colab (Linux) |
| Ollama install: `requires zstd` | `apt-get install zstd` first |
| Notebook ran before the MLflow server was ready | Wait for the server, then re-run the setup cell |
| Colab disconnects lose state | Optional Google Drive-backed SQLite store |

---

## Summary

<div class="cols">
<div>

### Prompt Registry — *what you say*
- Immutable versions with commit messages
- Aliases decouple prompts from deploys
- Diff, evaluate, promote, roll back

</div>
<div>

### Local inference — *how we ran it*
- Llama 3.2 (3B) via Ollama on a Colab T4
- OpenAI-compatible API, so code barely changes
- No API key or rate limit during evaluation

</div>
</div>

> **Take-away:** prompt changes become measurable, reversible and decoupled from deployment.

---

## References

- MLflow Prompt Registry — https://mlflow.org/docs/latest/genai/prompt-registry/
- Evaluate & monitor GenAI — https://mlflow.org/docs/latest/genai/eval-monitor/
- MLflow AI Gateway — https://mlflow.org/docs/latest/genai/governance/ai-gateway/
- Ollama — https://ollama.com/ &nbsp;·&nbsp; Llama 3.2 — https://ollama.com/library/llama3.2
- Pydantic AI — https://ai.pydantic.dev/
- Source repository — https://github.com/gromx-log/prompt_registry_and_ai_gateway_using_mlflow_and_local_ollama

---

<!-- _class: title -->
<!-- _paginate: false -->

# Thank you
## Questions & discussion
