# responsible-ai-system-design

Course project: an AI reading assistant that supports text comprehension with
on-demand, context-grounded help — without spoiling what the reader hasn't reached.
See `report.md` for the full write-up.

## Repository structure

```
report.md            Written report (problem, proposal, architecture, validation)
antispoiler/         Position-bounded anti-spoiler mechanism + eval harness (see its README)
validator/           validation layer: core pipeline + service bridge + dictionary
                       (design record in validator/DECISIONS.md, results in OBSERVATIONS.md)
app.py               Interactive demo (FastAPI): reading UI + validation
static/index.html    Demo frontend — select text, pick a feature, see the 3-way validation
notebooks/           antispoiler_eval.ipynb, validator_judge_poc_v2.ipynb (LLM-3 PoC),
                       validator_judge_poc_v3.ipynb (per-feature τ calibration)
docs/                Persona + user journey, and reference papers
links.md             Shared links (slides, etc.)
```

## Running the anti-spoiler eval

Apple Silicon needs a native arm64 conda env (the repo's base miniconda is x86):

```bash
CONDA_SUBDIR=osx-arm64 conda create -y -n antispoiler-arm python=3.11
conda run -n antispoiler-arm conda config --env --set subdir osx-arm64
conda run -n antispoiler-arm pip install -r requirements.txt
conda run -n antispoiler-arm python tests/test_logic.py   # no API key needed
```

Put `API_KEY` and `HF_TOKEN` in a `.env` at the repo root (git-ignored). Full
details in `antispoiler/README.md`.

## Running the interactive demo

The demo (`app.py` + `static/index.html`) is the reading assistant UI: select a
passage, click a feature (Define / Paraphrase / Contextualize / Recall), and the
generated answer is run through the LLM-3 validator and shown as Valid / Hedged /
Not reliable. The reader pane can switch between the extracted text view and a
PDF view; opening a local PDF makes that PDF the active in-memory document and
uses page number as the reader-position axis. Needs `ANTHROPIC_API_KEY` in the
repo-root `.env`.

```bash
conda run -n antispoiler-arm uvicorn app:app --port 8000
# then open http://127.0.0.1:8000
```

Launch from the same env you installed into (`antispoiler-arm`) — running from a
different Python is the usual cause of the `Dictionary: UNAVAILABLE` message below.
The first request is slow (it downloads the embedding model and indexes the book
once); opening a PDF also builds a temporary index for that document. The Define
feature needs the WordNet corpus (next section). Scanned/OCR-only PDFs are not
supported in this demo because the backend requires extractable text.

### Dev mode — cheap models via OpenRouter

For fast/cheap iteration, set `APP_MODE=dev` on the run command. This routes **both** the
generator and the validator through a single cheap model on
[OpenRouter](https://openrouter.ai) (OpenAI-compatible) instead of Anthropic. Add
`OPENROUTER_API_KEY` to the repo-root `.env`, then:

```bash
APP_MODE=dev conda run -n antispoiler-arm uvicorn app:app --port 8000
```

The startup banner shows the active mode:

```
Mode: DEV  |  backend=openrouter  generator=deepseek/deepseek-v4-pro  validator=deepseek/deepseek-v4-pro
```

The default dev model is `deepseek/deepseek-v4-pro`; override it without editing code via
`OPENROUTER_MODEL=<any OpenRouter model id>` (e.g. `OPENROUTER_MODEL=deepseek/deepseek-chat`).
Dev mode is for development only — it uses one model for both roles, which collapses the
validator≠generator separation (D13) and is **not** the characterized setup. **Prod is the
default** (no flag) and keeps Anthropic Haiku + Sonnet.

## Installing nltk WordNet corpus for definition validator


```bash
# One-time: download the WordNet corpus the definition validator grounds on.
# (nltk itself is installed by requirements.txt; only the data needs fetching.)
conda run -n antispoiler-arm python -m nltk.downloader wordnet omw-1.4
```

At startup the server prints a dictionary probe:

```
Dictionary: OK — wordnet (13 senses for 'test')
```

A missing dictionary is non-fatal — the definition validator just degrades every
term to **Hedged** ("couldn't ground this definition") — but then definitions are
never actually validated, so it's worth fixing.

**Troubleshooting `Dictionary: UNAVAILABLE`.** The message names the cause:

- `… ModuleNotFoundError: No module named 'nltk'` — the server is running in a
  Python that doesn't have nltk. This usually means deps were installed in one
  environment but the server was launched from another (e.g. installed into
  `antispoiler-arm` via `conda run`, but ran `python -m uvicorn` from a pyenv).
  **Install deps and launch the server in the same environment.** Either route
  everything through `conda run -n antispoiler-arm …`, or install into whatever
  `python` you launch with: `python -m pip install nltk` (or `-r requirements.txt`).
- `… corpus / Resource 'wordnet' not found` — nltk is installed but the WordNet
  data isn't; run the `nltk.downloader` command above. The corpus lives in
  `~/nltk_data` and is shared across all Python environments, so it only needs
  downloading once.

## Running the notebooks

`notebooks/validator_judge_poc_v2.ipynb` (the LLM-3 PoC) and
`validator_judge_poc_v3.ipynb` (per-feature τ calibration) import the live
`validator/` and `antispoiler/` packages, so the Jupyter **kernel must be the
`antispoiler-arm` env**. Simplest: install JupyterLab into that env and launch it
from there, so the default kernel is automatically `antispoiler-arm`:

```bash
conda run -n antispoiler-arm pip install jupyterlab
conda run -n antispoiler-arm jupyter lab
# then open notebooks/validator_judge_poc_v3.ipynb and Run All
```

Sanity check: the setup cell should print `validator : claude-sonnet-4-6` and
`dictionary: (True, …)`. A `ModuleNotFoundError: No module named 'validator'` means
the kernel is the wrong environment — launch `jupyter lab` from `antispoiler-arm`,
not from another env. (Alternatively, register the env as a kernel with
`conda run -n antispoiler-arm python -m ipykernel install --user --name antispoiler-arm`
and pick it from JupyterLab or VS Code.)
