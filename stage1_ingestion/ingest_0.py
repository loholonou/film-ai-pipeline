"""
Stage 1: Film Ingestion Pipeline
=================================
Parses a raw video file into structured, analyzable components:
  - Frame extraction (1 fps)
  - Audio separation
  - Dialogue transcription (Whisper)
  - Shot boundary detection (PySceneDetect)

Usage:
    python ingest.py --input path/to/film.mp4 --output path/to/output_dir

Requirements:
    pip install openai-whisper scenedetect[opencv] ffmpeg-python tqdm
    Also requires ffmpeg installed on your system:
        brew install ffmpeg
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Third-party imports — install with pip if missing
try:
    import whisper
except ImportError:
    sys.exit("Missing: pip install openai-whisper")

try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
except ImportError:
    sys.exit("Missing: pip install scenedetect[opencv]")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("Missing: pip install tqdm")


# ─────────────────────────────────────────────
# 1. FRAME EXTRACTION
# ─────────────────────────────────────────────

def extract_frames(video_path: Path, output_dir: Path, fps: float = 1.0) -> dict:
    """
    Extract frames from a video at a given frame rate using ffmpeg.

    Why ffmpeg? It's the industry standard for video processing, handles
    virtually every codec, and is extremely fast via hardware acceleration.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        fps: Frames per second to extract (default: 1 frame per second).

    Returns:
        dict with metadata about the extraction.
    """
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(frames_dir / "frame_%06d.jpg")

    print(f"\n[1/4] Extracting frames at {fps} fps...")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",          # Sample at target fps
        "-q:v", "2",                   # JPEG quality (2=high, 31=low)
        "-hide_banner",
        "-loglevel", "error",          # Suppress ffmpeg noise
        output_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed:\n{result.stderr}")

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    print(f"    ✓ Extracted {len(frame_files)} frames → {frames_dir}")

    return {
        "frames_dir": str(frames_dir),
        "fps": fps,
        "total_frames": len(frame_files),
        "sample_frame": str(frame_files[0]) if frame_files else None,
    }


# ─────────────────────────────────────────────
# 2. AUDIO SEPARATION
# ─────────────────────────────────────────────

def extract_audio(video_path: Path, output_dir: Path) -> dict:
    """
    Separate the audio track from the video.

    We export as 16kHz mono WAV — this is exactly what Whisper expects,
    so doing it here avoids Whisper having to re-decode the video itself.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save the audio file.

    Returns:
        dict with metadata about the extracted audio.
    """
    audio_path = output_dir / "audio.wav"

    print("\n[2/4] Extracting audio track...")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",                          # No video
        "-acodec", "pcm_s16le",         # Uncompressed 16-bit PCM
        "-ar", "16000",                 # 16kHz sample rate (Whisper standard)
        "-ac", "1",                     # Mono channel
        "-hide_banner",
        "-loglevel", "error",
        "-y",                           # Overwrite if exists
        str(audio_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")

    size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"    ✓ Audio extracted → {audio_path} ({size_mb:.1f} MB)")

    return {
        "audio_path": str(audio_path),
        "sample_rate": 16000,
        "channels": 1,
        "format": "WAV PCM 16-bit",
        "size_mb": round(size_mb, 2),
    }


# ─────────────────────────────────────────────
# 3. DIALOGUE TRANSCRIPTION (WHISPER)
# ─────────────────────────────────────────────

def transcribe_audio(audio_path: Path, output_dir: Path, model_size: str = "base") -> dict:
    """
    Transcribe dialogue using OpenAI Whisper.

    Why Whisper? It's open-source, runs locally, handles multiple languages,
    and outputs word-level timestamps — essential for aligning dialogue to shots.

    Model size tradeoffs (speed vs. accuracy):
        tiny   → fastest, lowest accuracy  (~1 GB VRAM)
        base   → good balance              (~1 GB VRAM)  ← default
        small  → better accuracy           (~2 GB VRAM)
        medium → near state-of-art         (~5 GB VRAM)
        large  → best accuracy             (~10 GB VRAM)

    On a MacBook without a GPU, 'base' or 'small' are recommended.

    Args:
        audio_path: Path to the extracted WAV file.
        output_dir: Directory to save the transcript.
        model_size: Whisper model size to use.

    Returns:
        dict with transcript metadata and path to JSON output.
    """
    transcript_path = output_dir / "transcript.json"

    print(f"\n[3/4] Transcribing dialogue (Whisper '{model_size}' model)...")
    print("    Note: First run downloads the model weights (~150MB for 'base')")

    model = whisper.load_model(model_size)

    # transcribe() returns: {"text": str, "segments": [...], "language": str}
    # Each segment has: id, start, end, text, tokens, confidence
    result = model.transcribe(
        str(audio_path),
        verbose=False,
        word_timestamps=True,   # Word-level timing — useful for later stages
    )

    # Save full transcript
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    segment_count = len(result.get("segments", []))
    language = result.get("language", "unknown")
    preview = result.get("text", "")[:200].strip()

    print(f"    ✓ Transcribed {segment_count} segments | Language: {language}")
    print(f"    Preview: \"{preview}...\"")

    return {
        "transcript_path": str(transcript_path),
        "language": language,
        "segment_count": segment_count,
        "text_preview": preview,
    }


# ─────────────────────────────────────────────
# 4. SHOT BOUNDARY DETECTION (PySceneDetect)
# ─────────────────────────────────────────────

def detect_shots(video_path: Path, output_dir: Path, threshold: float = 27.0) -> dict:
    """
    Detect shot boundaries using PySceneDetect's ContentDetector.

    Why ContentDetector vs ThresholdDetector?
    - ThresholdDetector: Looks for sudden changes in average pixel brightness.
      Simple but misses cuts between scenes of similar brightness.
    - ContentDetector: Analyzes changes in HSV colour space and motion
      across frames. More robust to complex real-world edits. ← we use this

    The threshold (default 27.0) controls sensitivity:
    - Lower value → more sensitive (detects more cuts, including soft ones)
    - Higher value → less sensitive (only hard cuts)
    - 27.0 is a good starting point for professionally edited film.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save the shots CSV.
        threshold: Detection sensitivity (27.0 recommended for film).

    Returns:
        dict with shot metadata and path to CSV output.
    """
    shots_path = output_dir / "shots.csv"

    print(f"\n[4/4] Detecting shot boundaries (threshold={threshold})...")

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    # Write CSV: shot_id, start_time, end_time, start_frame, end_frame, duration
    with open(shots_path, "w", encoding="utf-8") as f:
        f.write("shot_id,start_time_s,end_time_s,start_frame,end_frame,duration_s\n")
        for i, (start, end) in enumerate(scene_list):
            f.write(
                f"{i},"
                f"{start.get_seconds():.3f},"
                f"{end.get_seconds():.3f},"
                f"{start.get_frames()},"
                f"{end.get_frames()},"
                f"{(end - start).get_seconds():.3f}\n"
            )

    print(f"    ✓ Detected {len(scene_list)} shots → {shots_path}")

    # Compute basic shot statistics
    if scene_list:
        durations = [(end - start).get_seconds() for start, end in scene_list]
        avg_duration = sum(durations) / len(durations)
        print(f"    Average shot duration: {avg_duration:.1f}s")
    else:
        avg_duration = 0

    return {
        "shots_path": str(shots_path),
        "total_shots": len(scene_list),
        "avg_shot_duration_s": round(avg_duration, 2),
        "threshold_used": threshold,
    }


# ─────────────────────────────────────────────
# MAIN PIPELINE RUNNER
# ─────────────────────────────────────────────

def run_pipeline(video_path: str, output_dir: str, whisper_model: str = "base") -> dict:
    """
    Run all four ingestion stages and write a manifest.json summary.

    Args:
        video_path: Path to the input film file (.mp4, .mkv, .mov, etc.)
        output_dir: Root directory for all outputs.
        whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').

    Returns:
        manifest dict with metadata from all stages.
    """
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        sys.exit(f"Error: Video file not found: {video_path}")

    print("=" * 55)
    print("  STAGE 1: Film Ingestion Pipeline")
    print("=" * 55)
    print(f"  Input:  {video_path.name}")
    print(f"  Output: {output_dir}")
    print("=" * 55)

    manifest = {
        "input_file": str(video_path),
        "output_dir": str(output_dir),
        "film_name": video_path.stem,
    }

    # Run each stage and collect metadata
    manifest["frames"]     = extract_frames(video_path, output_dir)
    manifest["audio"]      = extract_audio(video_path, output_dir)
    manifest["transcript"] = transcribe_audio(output_dir / "audio.wav", output_dir, whisper_model)
    manifest["shots"]      = detect_shots(video_path, output_dir)

    # Write manifest.json — this is the "index" for downstream pipeline stages
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 55)
    print("  ✅ Ingestion complete!")
    print(f"  Manifest → {manifest_path}")
    print("=" * 55)
    print(f"\n  Output structure:")
    print(f"  {output_dir}/")
    print(f"  ├── frames/          ({manifest['frames']['total_frames']} frames)")
    print(f"  ├── audio.wav        ({manifest['audio']['size_mb']} MB)")
    print(f"  ├── transcript.json  ({manifest['transcript']['segment_count']} segments)")
    print(f"  ├── shots.csv        ({manifest['shots']['total_shots']} shots)")
    print(f"  └── manifest.json    (pipeline index)")
    print()

    return manifest


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: Film Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ingest.py --input film.mp4 --output ./output
    python ingest.py --input film.mp4 --output ./output --whisper-model small
        """
    )
    parser.add_argument("--input",          required=True,  help="Path to input video file")
    parser.add_argument("--output",         required=True,  help="Path to output directory")
    parser.add_argument("--whisper-model",  default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")

    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.whisper_model)
