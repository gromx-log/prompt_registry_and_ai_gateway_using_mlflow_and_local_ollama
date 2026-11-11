# Prompt Registry with MLflow 3.13 and Local Llama 3.2 (Google Colab Pro)

[![MLflow](https://img.shields.io/badge/MLflow-3.13-blue.svg)](https://mlflow.org/)
[![Model](https://img.shields.io/badge/Model-Llama_3.2_3B-green.svg)](https://ollama.com/library/llama3.2)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

A hands-on demo of the **MLflow 3.13 Prompt Registry** (versioning, aliases, evaluation) where the LLM is **Llama 3.2 (3B) served by Ollama on a Google Colab Pro GPU** instead of a paid cloud API.

It is adapted from the MLflow 3.13 *Prompt Registry & AI Gateway* workshop. **What we changed:** the OpenAI/Claude calls are replaced by a local, OpenAI-compatible Ollama endpoint, so no API key and no provider rate limit is involved.

---

## Why we changed the model backend

- No OpenAI / Claude key was available.
- The Gemini free tier returned **HTTP 429** (limit of 5 requests/minute per model), which breaks evaluation loops that fire many calls at once.
- `mlflow gateway start` is **not supported on Windows**.

Running Ollama on a Colab T4 solved all three: the `openai` SDK code stays the same, only `base_url` and `model` change.

## What the notebook does

1. Checks the GPU, installs Ollama and pulls `llama3.2:3b`.
2. Starts an MLflow tracking server (SQLite backend) inside Colab.
3. **Prompt Registry:** registers `v1` and `v2`, prints a Git-style diff, promotes `v2` with the `@production` alias.
4. Classifies support tickets with the governed prompt (OpenAI SDK -> Ollama).
5. Returns a typed `TriageResult(category, reason)` with **Pydantic AI**.
6. Evaluates prompt versions with `mlflow.genai.evaluate()` on a 4-row dataset.
7. Registers `v3` (written by hand), promotes it to `@production` and re-evaluates.

## Results (from our Colab run)

| Version | `category_match` | `label_guidelines` | Note |
| :---: | :---: | :---: | --- |
| v1 | 0.50 | 1.00 | minimal prompt |
| v2 | 1.00 | 1.00 | explicit labels and rules, promoted to `@production` |
| v3 | 1.00 | 1.00 | `@production` re-evaluated |

### Limitations

- **Tiny dataset** (4 rows, one per label): a demo, not a statistical result. v3 cannot be told apart from v2 on it.
- **Lenient scorer:** `category_match` accepts the expected label anywhere in the output.
- **v3 is hand-written.** We did not run `mlflow.genai.optimize_prompts()`.
- **No AI Gateway in this run.** Gateway features (routing, fallbacks, usage tracking) are covered conceptually in the slides; the notebook calls Ollama directly.
- The MLflow SQLite DB lives in the Colab session and is lost on disconnect unless stored on Google Drive.

---

## Repository contents

```text
.
├── mlflow_3_demo_colab.ipynb               # the executable Colab notebook
├── mlflow_3_demo_colab.ipynb - Colab.pdf   # executed run exported from Colab
├── slides.md                               # Marp source of the presentation (UNTAR template)
├── slides.html                             # built slides (needs the assets/ folder next to it)
├── slides.pdf                              # built slides, ready to send
├── assets/untar_template.png               # slide background (UNTAR footer / logo)
├── requirements.txt
├── .gitignore
├── LICENSE                                 # GPL-3.0 (inherited from the upstream workshop)
└── README.md
```

## Run it (Google Colab Pro, nothing local)

1. Open <https://colab.research.google.com/> -> **Upload** -> `mlflow_3_demo_colab.ipynb`.
2. **Runtime -> Change runtime type -> T4 GPU -> Save**.
3. **Runtime -> Run all.**
4. In the MLflow-server cell, click the generated link to open the MLflow UI (Prompts and Experiments tabs).

> If the setup cell prints "connection refused" warnings, the MLflow server was still starting. Wait a few seconds and re-run the setup cell.

## Build / export the slides

The deck is written in [Marp](https://marp.app/). Requires Node.js and Chrome or Edge.

```bash
# HTML
npx @marp-team/marp-cli@latest --no-stdin slides.md -o slides.html --allow-local-files

# PDF (send this one to the lecturer)
npx @marp-team/marp-cli@latest --no-stdin slides.md --pdf -o slides.pdf --allow-local-files
```

Alternative: install the **Marp for VS Code** extension, open `slides.md`, then **Command Palette -> "Marp: Export Slide Deck..." -> PDF**.

## Technologies

[MLflow 3.13](https://mlflow.org/) · [Ollama](https://ollama.com/) · [Llama 3.2 3B](https://ollama.com/library/llama3.2) · [Pydantic AI](https://ai.pydantic.dev/) · [Google Colab](https://colab.research.google.com/)

## License

GPL-3.0, see [LICENSE](LICENSE). It is inherited from the upstream workshop this project is adapted from; check the upstream repository before changing it.
