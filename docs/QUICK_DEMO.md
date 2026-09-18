# How to Use StoryForge-RAG

This guide walks you through the full process in plain language — from setup to generating your first story.

---

## What does this do?

StoryForge-RAG takes a question or topic (your **query**), searches through a library of stored stories, picks out relevant facts, and writes a brand new **5-section story** grounded in those facts.

It does not make things up. Every named character, place, and event in the output comes directly from your story library.

---

## Before you start — one-time setup

### 1. Start Ollama (the AI that writes the story)

```powershell
docker compose up -d
docker exec -it ollama ollama pull qwen3.5:9b
```

This downloads the writing model (~6 GB, first run only). Wait until it finishes.

### 2. Install Python packages

```powershell
pip install -r requirements.txt
```

### 3. Create your config file

```powershell
copy setup.example.yaml setup.yaml
```

Open `setup.yaml` and set:
- `BASE_PATH` → the full path to this project folder
- `facehugging_api` → your Hugging Face API key (free at huggingface.co)

### 4. Start the API

```powershell
python main.py
```

Open your browser at **http://localhost:8000/docs** — you should see the API explorer.

---

## Step A — Add stories to the library

Stories must be plain `.txt` files, one story per file, saved in `data/stories/`.
If you just extracted a PDF/EPUB, clean and split it first — see [`DATA_PREP.md`](./DATA_PREP.md).

Once your files are there, run these three scripts in order:

```powershell
# 1. Prepare and enrich the stories (adds tags, summaries, chunking)
py scripts/step1_prepare_and_enrich.py

# 2. Build the ingest list
py scripts/records_to_ingest_manifest.py

# 3. Load everything into the search database
py scripts/ingest_manifest.py
```

You only need to do this again when you add or change stories.

> **Starting fresh?** Run `py scripts/reset_and_ingest.py` to wipe and reload everything in one step.

---

## Step B — Generate a story

### Option 1: Use the browser (easiest)

Go to **http://localhost:8000/docs**, find `POST /create-eval/story_generate`, click **Try it out**, and paste this:

```json
{
  "query": "A soldier returns home to find everything has changed",
  "mode": "fast",
  "length": "medium",
  "save": true
}
```

Click **Execute**. The story appears in the response.

---

### Option 2: Use the terminal

```powershell
curl -s -X POST http://localhost:8000/create-eval/story_generate `
  -H "Content-Type: application/json" `
  -d '{"query": "A soldier returns home to find everything has changed", "mode": "fast", "length": "medium", "save": true}'
```

---

### Option 3: Stream the story as it writes (word by word)

```powershell
curl -N -X POST http://localhost:8000/orchestration/generate_stream `
  -H "Content-Type: application/json" `
  -d '{"query": "A soldier returns home", "mode": "fast", "length": "medium"}'
```

Tokens stream in real time as the model writes.

---

## Choosing story length and style

### Length

| Value | Words | Narration time |
|-------|-------|----------------|
| `"short"` | ~450 | ~3 minutes |
| `"medium"` | ~900 | ~6 minutes |
| `"long"` | ~1500 | ~11 minutes |
| `"epic"` | ~2200 | ~16 minutes |
| `"10min"` | auto | exactly 10 minutes |
| `"1800"` | 1800 | ~13 minutes |

Leave `length` out and the system picks a default based on `mode`.

### Mode

| Value | What it does |
|-------|-------------|
| `"fast"` | Quick generation, good quality |
| `"thinking"` | Slower, more detailed — better for longer stories |

---

## Story structure

Every generated story has exactly **5 sections**:

```
[SECTION 1] Who, Where, When — The Setup
[SECTION 2] The Inciting Incident
[SECTION 3] Rising Action
[SECTION 4] Climax / Confrontation
[SECTION 5] Resolution / Outcome
```

The system checks that all 5 sections are present and that the story is long enough. If not, it automatically tries to improve the draft before returning it to you.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Story comes back empty | Rare now — thinking mode retries once with fast sampling; length-guard / agentic loop keep the best draft they have. If it still fails, use `"mode": "fast"` |
| "No documents found" | Run the ingest scripts (Step A) first. If you just extracted a book, clean it first ([`DATA_PREP.md`](./DATA_PREP.md)) |
| API not responding | Check that `python main.py` is still running |
| Ollama errors | Run `docker compose up -d` to restart Ollama |
| HF rate limits on evaluation | Set `Evaluation_mode: "local"` in `setup.yaml` (CPU by default) |

---

## Quick reference — all story endpoints

| Endpoint | What it does |
|----------|-------------|
| `POST /create-eval/story_generate` | Generate + evaluate (recommended) |
| `POST /orchestration/run_step` + `4_generate_story_3step` | Single-pass, no evaluation |
| `POST /orchestration/run_step` + `4_generate_story_agentic` | With refine/retry loop |
| `POST /orchestration/generate_stream` | Stream tokens as they generate |

---

## After changing config or prompts

Restart the API after any edits to `setup.yaml` or `prompts.yaml`:

```powershell
# Stop with Ctrl+C, then:
python main.py
```

Config and prompts are cached on startup.
