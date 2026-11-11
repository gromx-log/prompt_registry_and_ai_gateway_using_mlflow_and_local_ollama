# TODO — remaining items (for you or Gemini)

The slides (`slides.md` -> `slides.html` / `slides.pdf`) are already built on the UNTAR template.
These items need information only you have, or are optional polish.

## Must do before sending to the lecturer
- [ ] **Slide 1:** Remove `[Course name]`, `[Lecturer name]`, `[Your name] — [NIM]` in `slides.md`.
- [ ] **Slide 22 (References):** replace `[add your GitHub URL here]` after the repo is published.
- [ ] Rebuild the HTML and PDF after every edit to `slides.md` (commands are in `README.md`).
- [ ] Open `slides.pdf` once and flip through it: nothing may touch the dark footer bar.

## Recommended (makes the deck stronger)
- [ ] **Screenshots from your Colab run** (the MLflow UI link is in the server cell). Add them to `assets/` and use `![w:900](assets/name.png)`:
  - Prompts page showing v1, v2, v3 and the `production` alias
  - Prompt **Compare** view (v1 vs v2)
  - Experiments page with the `eval-v1-local` / `eval-v2-local` runs
- [ ] Add a screenshot of `nvidia-smi` (Tesla T4) to the *Architecture* or *Setup* slide.
- [ ] Re-run the notebook once more and confirm the numbers on the *Results* slide still match (LLM output can vary).

## Optional improvements to the project itself
- [ ] Enlarge the evaluation dataset (e.g. 20–30 tickets, several per label, some ambiguous) so v2 vs v3 can actually be compared.
- [ ] Make `category_match` strict (exact match after `strip().upper().strip(".")`).
- [ ] Try `mlflow.genai.optimize_prompts()` with `MetaPromptOptimizer` using the local model, then update the slides to report it honestly.
- [ ] Fix the MLflow "connection refused" warnings in the setup cell by waiting until `http://localhost:5000/health` answers before `mlflow.set_experiment(...)`.
- [ ] Persist the MLflow DB to Google Drive (`PERSIST_TO_DRIVE` option) if you want prompts to survive a disconnect.

## Repository hygiene
- [ ] Confirm the license: `LICENSE` is GPL-3.0 (copied from the upstream workshop). Keep it unless the upstream says otherwise.
- [ ] `git init`, commit, push (see the commands in the chat / README).

