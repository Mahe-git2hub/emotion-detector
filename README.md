# Real-time Emotion Detector for Video Meetings

The **Emotion Engagement Monitor** is a Python-based tool that analyses facial
expressions during live or recorded video calls. It highlights dominant emotions,
computes an engagement score, and optionally logs metrics for later review. The
application is designed to help meeting facilitators understand participation
levels and adapt communication strategies in real time.

## Key features

- **Live facial emotion analysis** powered by the
  [`fer`](https://github.com/justinshenk/fer) deep learning model.
- **Engagement scoring** that classifies emotions into positive, neutral, and
  negative buckets and smooths results with an exponential moving average.
- **Customisable emotion weights** allowing teams to redefine what counts as an
  engaged response.
- **OpenCV overlay** that annotates faces with their dominant emotions and a
  rolling engagement dashboard.
- **Headless & logging modes** for server environments or asynchronous analysis.

## Getting started

### Prerequisites

- Python 3.9 or later (the FER model depends on modern versions of `tensorflow`).
- A webcam or access to a video file/stream if you want to analyse live input.
- System packages required by OpenCV (Linux systems may need
  `libsm6`, `libxext6`, and `libxrender1`).
- MoviePy 1.x is installed via requirements; reinstall with `pip install -r requirements.txt` if a previous environment pulled MoviePy 2.x because FER still imports `moviepy.editor`.

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Running the monitor

```bash
# Analyse the default webcam
python -m emotion_detector.main --source 0

# Analyse a recorded meeting
python -m emotion_detector.main --source path/to/meeting.mp4

# Run without a preview window and log metrics to CSV
python -m emotion_detector.main --source 0 --no-display --log-file logs/session.csv
```

The interface opens an OpenCV window titled **Emotion Engagement Monitor**.
Press `q` or `Esc` to stop the session. When `--no-display` is set, the tool runs
headlessly and prints periodic summaries to the console.

## Understanding the metrics

Each detected face contributes to an engagement score between 0 and 1. Emotions
configured as **positive** contribute `1.0`, **neutral** emotions contribute
`0.5`, and **negative** emotions contribute `0.0`. The final score for a frame is
the average contribution across all visible faces. The application also keeps a
rolling average and an exponential moving average (smoothing factor 0.2 by
default) to reduce volatility.

You can customise the emotion categories using command-line arguments:

```bash
python -m emotion_detector.main \
  --positive-emotions "happy,surprise" \
  --neutral-emotions "neutral,calm" \
  --negative-emotions "sad,angry,disgust,fear"
```

The configuration is case insensitive, and unspecified categories fall back to
the defaults defined in `emotion_detector/config.py`.

## Logging output

Providing `--log-file path/to/file.csv` creates a CSV log with timestamps,
per-frame engagement statistics, and the number of faces detected. The log can
be imported into analytics tools to review participation trends after the
meeting.

## Project structure

```
emotion_detector/
├── analyzer.py          # Wrapper around the FER emotion detector
├── config.py            # Configuration dataclasses for emotion weights
├── engagement.py        # Engagement score calculation and smoothing
├── main.py              # Command-line entry point
├── visualizer.py        # Rendering of bounding boxes and dashboards
requirements.txt          # Python dependencies
```

## Development tips

- The FER model downloads weights the first time it runs; allow a few seconds on
  the first execution.
- For automated tests or CI where a webcam is unavailable, use `--max-frames`
  with a recorded sample video to limit execution time.
- The OpenCV preview window is optional; if the display fails (common in
  headless servers), the application will automatically fall back to console
  summaries.

## Disclaimer

Emotion recognition from facial expressions has limitations and can be biased by
lighting, camera position, and cultural factors. Use the generated insights as a
supporting signal rather than the sole indicator of participant engagement.
