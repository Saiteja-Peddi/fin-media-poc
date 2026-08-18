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

Pick **1** to ingest a media file, then **3** to ask questions about it. That's it.

See [Setup](#setup) for details and [Usage](#usage) for the full menu.

## How it works

```
ingest:  video/audio ──► 16kHz WAV ──► word-level transcript ──► ~30s chunks ──► embeddings ──► Chroma
         (ffprobe/         (WhisperX          (fixed window)      (Ollama         (vector store)
          ffmpeg)           distil-large-v3)                       mxbai-embed-large)

ask:     question ──► embedding ──► nearest chunks ──► ranked results (MM:SS ranges) + playable clips
```

## Setup

Run the setup script from the project root:

```bash
./setup.sh
```

It automates everything it safely can and prints a readiness report:

- creates (or reuses) the `.venv` virtualenv
- installs the Python dependencies from `requirements.txt`
- pulls the Ollama embedding model (`mxbai-embed-large`) if Ollama is running

It **can't** install system-level tools for you, so if any are missing it reports
them with the exact command to fix, then you re-run `./setup.sh`:

- **ffmpeg** — `brew install ffmpeg`
- **Ollama** — install from <https://ollama.com/download>, then start it with
  `ollama serve`

When the summary shows `All set`, you're ready to go.

> **Notes**
> - Use `.venv/bin/python` explicitly to run the app. On this machine a bare
>   `python` resolves to the system Python, not the project venv.
> - The first ingest run downloads the WhisperX transcription model
>   (`distil-large-v3`, ~1.5GB) and the alignment model (~360MB) automatically;
>   TLS certificates for those downloads are configured in `src/config.py`, so
>   no manual setup is needed.

## Usage

Everything runs through one interactive program — start it with no arguments and
follow the prompts:

```bash
.venv/bin/python app.py
```

It presents a menu and prompts you for the file path or question based on what you
pick:

```
What would you like to do?
  1) Transcribe / ingest a file
  2) Process new files in data/input
  3) Ask a question
  4) Quit
Select an option [1-4]: 1
Path to the video/audio file (any location on your system): data/input/your_file.mp4

  Detected media kind: video
  Extracting audio...
  Transcribing (first run downloads the model and is slow)...
    42 words transcribed.
    3 chunks produced.
  Stored 3 chunks from 'your_file.wav'.
```

- **Option 1** ingests a single file — paste any full path on your system. The
  file's type (video vs. audio) is detected from its actual contents, not its
  extension, so a misnamed or corrupt file is caught up front.
- **Option 2** scans `data/input` and ingests every file that isn't already
  indexed (a quick way to process a folder of downloads).

Then choose **option 3** to search:

```
Select an option [1-4]: 3
Your question: what did they say about gold?
How many results? [3]:

Results for: "what did they say about gold?"

[1] 00:08-00:12  (your_file.wav)
    Gold prices surged to record highs as investors sought safe haven assets
```

Each search also cuts a short playable clip for every result into `data/clips`.
The menu loops until you choose **4** (or type `q`). If nothing matches, it prints
`No matches found.`

## Project layout

```
setup.sh           One-shot environment setup + readiness check
app.py             The app: interactive menu for ingesting files and asking questions
src/
  config.py        Paths, model names, toggle flags, TLS setup
  models.py        Model wrappers: asr() (WhisperX), embed() (Ollama)
  ingest.py        detect_media_kind(), extract_audio(), chunk_words()
  store.py         ChromaDB persistent store: add_chunks(), search()
  query.py         ask() — searches the index and cuts a clip per result
  clip.py          cut_clip(), clear_clips_folder() — playable clips from results
data/
  input/           source video/audio files
  media/           extracted WAVs + cached transcripts
  clips/           short clips cut around each search hit
db/chroma/         persistent vector store
```
