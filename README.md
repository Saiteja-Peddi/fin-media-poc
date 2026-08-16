# fin-media-poc

A local proof-of-concept that makes spoken financial content **searchable by meaning**.
Point it at a video or audio file; it transcribes the speech with word-level
timestamps, splits it into time-stamped chunks, embeds them, and stores them in a
local vector database. You can then ask natural-language questions and get back the
matching moments with their time ranges.

Everything runs locally — no cloud, no API keys.

## Quick start

**1. Install the system tools** (once, if you don't already have them):

| Tool | Install |
|------|---------|
| Python 3.11 | <https://www.python.org/downloads/> |
| ffmpeg | `brew install ffmpeg` |
| Ollama | <https://ollama.com/download>, then run `ollama serve` |

**2. Get the code and set it up:**

```bash
git clone https://github.com/Saiteja-Peddi/fin-media-poc.git
cd fin-media-poc
./setup.sh
```

`./setup.sh` creates the virtualenv, installs Python dependencies, and pulls the
embedding model. It prints a checklist — when you see `All set`, you're ready. If it
flags a missing system tool, install it (table above) and re-run `./setup.sh`.

**3. Run it:**

```bash
.venv/bin/python app.py
```

Pick **1** to ingest a media file, then **2** to ask questions about it. That's it.

See [Setup](#setup) for details and [Usage](#usage) for the scripted CLIs.

## How it works

```
ingest:  video/audio ──► 16kHz WAV ──► word-level transcript ──► ~30s chunks ──► embeddings ──► Chroma
         (ffmpeg)          (WhisperX)          (fixed window)     (Ollama)      (vector store)

ask:     question ──► embedding ──► nearest chunks ──► ranked results with MM:SS time ranges
```

## Setup

Run the setup script from the project root:

```bash
./setup.sh
```

It automates everything it safely can and prints a readiness report:

- creates (or reuses) the `.venv` virtualenv
- installs the Python dependencies from `requirements.txt`
- pulls the Ollama embedding model (`nomic-embed-text`) if Ollama is running

It **can't** install system-level tools for you, so if any are missing it reports
them with the exact command to fix, then you re-run `./setup.sh`:

- **ffmpeg** — `brew install ffmpeg`
- **Ollama** — install from <https://ollama.com/download>, then start it with
  `ollama serve`

When the summary shows `All set`, you're ready to go.

> **Notes**
> - Use `.venv/bin/python` explicitly to run the app. On this machine a bare
>   `python` resolves to the system Python, not the project venv.
> - The first ingest run downloads the WhisperX alignment model (~360MB)
>   automatically; TLS certificates for that download are configured in
>   `src/config.py`, so no manual setup is needed.

## Usage

The easiest way to use the POC is the **interactive menu**. There are also two
scripted CLIs if you prefer passing arguments directly (e.g. for automation).

### Interactive menu (recommended)

Start it with no arguments:

```bash
.venv/bin/python app.py
```

It presents a menu and prompts you for the file path or question based on what you
pick:

```
What would you like to do?
  1) Transcribe / ingest a file
  2) Ask a question
  3) Quit
Select an option [1-3]: 1
Path to the video/audio file: data/input/your_file.mp4

Extracting audio...
Transcribing (first run downloads the model and is slow)...
  42 words transcribed.
  3 chunks produced.

Stored 3 chunks from 'your_file.wav' in the index.
```

Then choose option 2 to search:

```
Select an option [1-3]: 2
Your question: what did they say about gold?
How many results? [3]:

Results for: "what did they say about gold?"

[1] 00:08-00:12  (your_file.wav)
    Gold prices surged to record highs as investors sought safe haven assets

[2] 00:12-00:16  (your_file.wav)
    amid market uncertainty. Management raised full-year guidance and announced
```

The menu loops until you choose **3** (or type `q`). If nothing matches, it prints
`No matches found.`

## Project layout

```
setup.sh           One-shot environment setup + readiness check
app.py             Interactive menu: ingest a file or ask a question
run_ingest.py      Scripted CLI: ingest a media file end to end
ask.py             Scripted CLI: search the index by question
src/
  config.py        Paths, model names, toggle flags, TLS setup
  models.py        Model wrappers: asr() (WhisperX), embed() (Ollama)
  ingest.py        extract_audio(), chunk_words()
  store.py         ChromaDB persistent store: add_chunks(), search()
  query.py         ask() — question search entry point
  clip.py          (not yet implemented) cut playable clips from results
data/
  input/           source video/audio files
  media/           extracted WAVs + cached transcripts
db/chroma/         persistent vector store
```
