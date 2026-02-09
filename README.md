## Audio-driven sprite-based video pipeline

This project implements a deterministic, audio-driven talking-head video
generation pipeline with the following properties:

- **Language**: Python 3.10
- **FPS**: fixed at 8
- **Audio format**: mono, float32, 22050 Hz
- **Time units**: seconds (float) everywhere
- **Rendering**: sprite-based (no ML models)
- **Muxing**: `ffmpeg` for H.264 + AAC in MP4

### High-level steps

- **Audio ingestion** (`audio/ingest.py`): validate and normalize a mono, float32
  22050 Hz WAV buffer already held in memory and return an `AudioBuffer`.
- **Phoneme extraction** (`alignment/phonemes.py`): define the interface for a
  forced aligner (e.g. MFA, Gentle) that returns ARPAbet phonemes with
  timestamps. The function raises `NotImplementedError` by default with a TODO
  comment indicating where to integrate a real aligner.
- **Phoneme → Viseme mapping** (`visemes/map.py`): map ARPAbet phonemes to a
  small, static set of viseme names.
- **Viseme smoothing** (`visemes/smooth.py`): enforce a minimum viseme duration,
  merge adjacent identical visemes, and slightly stretch plosive visemes.
- **Frame rendering** (`render/renderer.py`): use Pillow to overlay viseme
  sprites and a blink sprite at 8 FPS, adding small deterministic vertical
  jitter and random (but seeded) blinks.
- **Muxing** (`mux/ffmpeg.py`): pipe raw RGB frames and int16 PCM audio into
  `ffmpeg`, using a named pipe for audio, to produce an MP4 file.
- **Orchestration** (`main.py`): wire everything end-to-end for a single input
  audio file and sprites directory, performing aggressive sanity checks.

### Running the pipeline

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare sprites under `render/sprites/`. At minimum you should provide:

   - `REST.png` or `base.png` or `neutral.png` — neutral face/base sprite
   - `blink.png` — blink overlay sprite
   - Optional viseme sprites such as `PBM.png`, `FV.png`, `TH.png`, `SZ.png`,
     `CHJ.png`, `KNG.png`, `Y.png`, `UW.png`, `AA.png`, `IY.png`, `ER.png`,
     `L.png`, etc.

   All sprites should be the same resolution and will be composited in RGBA.

3. Integrate a real forced aligner in `alignment/phonemes.py` by implementing
   `extract_phonemes` to call MFA, Gentle, or another deterministic aligner and
   return a sorted list of `Phoneme` objects.

4. Run the pipeline:

   ```bash
   python -m main input.wav render/sprites output.mp4
   ```

   - `input.wav` must be a mono float32 WAV at 22050 Hz.
   - The script will assert aggressively on invalid inputs or unexpected
     intermediate states and will fail fast instead of silently degrading
     behavior.

