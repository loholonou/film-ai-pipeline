# Film AI Pipeline

An ML pipeline for understanding film language and generating trailers.

## Pipeline Stages
- **Stage 1 — Ingestion:** Frame extraction, audio separation, transcription, shot detection
- Stage 2 — Shot Understanding *(coming soon)*
- Stage 3 — Narrative Structure *(coming soon)*

## Setup
```bash
conda activate film-ai
pip install openai-whisper "scenedetect[opencv]" librosa tqdm ffmpeg-python
```

## Usage
```bash
python stage1_ingestion/ingest.py --input film.mp4 --output ./output
```
