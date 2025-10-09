# Youth Impact voice evaluation harness

Youth Impact is an NGO focused on strengthening children's foundational math through teacher-lead, phone-based tutoring. By 2026, they aim to reach 2.5 million across Karnataka, India.

## Background

To improve cost-effectiveness, they are working to automate the final math-level assessment using an ASR model.

However, off-the-shelf ASR don't perform well with children's voices in low-bandwidth settings. To address this, Youth Impact is building a high-quality dataset of children's voice-transcription pairs for structured evaluation tasks such as selecting the best STT/TTS off-the-shelf or custom system, testing noise-reduction strategies, and fine-tuning models. They already have a large repository (9 hours) of call recordings and now need Kannada transcribers to chunk, transcribe, and label children's voices.

## Dataset Information

**Total audio:** 9.01 hours across 41 files (177 MB)

**Key Statistics:**

- Duration range: 10 seconds to 32 minutes (avg: 13.2 min)
- Sample rates: Primarily 8kHz (56%), with 48kHz (17%), 16kHz (15%), 44.1kHz (12%)
- Codecs: AAC (56%), AMR-NB (20%), MP3 (20%), PCM (5%)
- Channels: 98% mono, 2% stereo
- Bit rates: 12.4 to 160 kbps (avg: 55 kbps)
- Content: Tutor and child voices (not diarized in source recordings)

This represents real-world, low-bandwidth phone call audio - the exact conditions the production ASR system must handle.

## Ground Truth Strategy

⚠️ **IMPORTANT CAVEAT:** This evaluation uses a **two-phase approach** due to timing constraints.

### Phase 1: Whisper as Temporary Baseline (Current)

In the absence of professional human transcriptions, we use **Whisper Large V3** as a pseudo-ground-truth baseline. This allows us to:

- Begin evaluation immediately rather than waiting
- Generate comparative metrics between STT models
- Establish relative performance rankings

**Critical limitation:** Results show **relative similarity to Whisper**, NOT absolute accuracy. A model scoring better than Azure may mean it's "more Whisper-like" rather than "more accurate."

### Phase 2: Professional Transcriptions (Target)

**URGENCY:** Professional Kannada transcriptions are essential for valid evaluation. Without them, we cannot:

- Determine true accuracy of any model (including Whisper)
- Make confident production deployment decisions
- Understand model performance on actual children's Kannada speech

**Timeline:** Professional transcriptions are in progress via transcription service. Upon delivery:

1. Re-evaluate ALL models (including Whisper) against human ground truth
2. Compare Phase 1 vs Phase 2 rankings to validate/invalidate initial findings
3. Generate authoritative accuracy metrics for production decisions

**Deliverable format:** SRT files with timestamps and speaker labels (tutor/child)

## STT Models Under Evaluation

Based on Kannada language support and performance with low-bandwidth audio:

**Tier 1 (MVP):**

1. **Whisper Large V3** - OpenAI's multilingual model (Phase 1 baseline, Phase 2 evaluated)
2. **Azure Speech Services** - Kannada (kn-IN) confirmed, strong India presence
3. **AssemblyAI** - Explicit Kannada support across 99+ languages

**Tier 2 (If validated):** 4. **Google Cloud Speech-to-Text V2** - Likely Kannada support (pending verification)

**Excluded:**

- Deepgram (no Kannada support, only Hindi/Indian English)

**Evaluation criteria:**

- Word Error Rate (WER) / Character Error Rate (CER)
- Cost per hour of audio
- Processing latency
- Custom WER for math-critical vocabulary (numbers, terms)

**Budget estimate:** ~$25-30 for full 9-hour evaluation (4 models)

## Architecture

This codebase contains exploratory code for a voice model evaluation harness built with Python 3.11 and UV. It computes standard speech evaluation metrics (WER, CER) and provides utilities for audio processing, transcription generation, and results analysis.

### Project Structure

```
/
  pyproject.toml           # UV project definition
  uv.lock                  # Locked dependencies
  .python-version          # Python 3.11
  .env                     # Environment variables (gitignored)

  /src
    /yi_voice_eval         # Core package
      __init__.py
      audio_processing.py  # Preprocessing, resampling, normalization
      transcription.py     # STT model integrations (Whisper, Azure, etc.)
      evaluation.py        # WER, CER, custom metrics
      storage.py           # fsspec abstraction (local/cloud)
      sheets_output.py     # Google Sheets API integration
      format_parsers.py    # SRT, Word, Whisper JSON parsers

  /notebooks               # Modular Jupyter notebooks
    01_baseline_whisper.ipynb           # Generate Whisper baseline
    02_stt_comparison.ipynb             # Run competing STT models
    03_results_analysis.ipynb           # Aggregate metrics, output to Sheets

  /data                    # Local storage (gitignored)
    /raw_audio             # Original 41 files
    /processed             # Standardized audio (16kHz, mono, WAV)
    /transcriptions        # Model outputs (SRT format)
      /whisper_baseline
      /azure_speech
      /assemblyai
      /ground_truth        # Professional transcriptions (when available)

  /config
    config.template.yaml   # Template configuration (committed)
    config.yaml            # Actual configuration (gitignored)

  /tests                   # Unit tests for core modules

  README.md
```

### Storage Abstraction

Uses **fsspec** for seamless local/cloud storage:

- **Development:** Local filesystem (`/data` directory)
- **Production:** Google Cloud Storage or other cloud providers
- Config-driven: `storage_backend: local` or `storage_backend: gcs`
- Migration path: Simple config change + data sync

### Design Principles

1. **Modularity:** Core logic in `/src`, notebooks stay lightweight
2. **Version Control:** All code committed, credentials/data gitignored
3. **Reproducibility:** Locked dependencies, config snapshots in results
4. **Testability:** Preprocessing and evaluation logic unit-tested
5. **Evolvability:** Designed to evolve from exploration → API → product

## Setup

### Prerequisites

- Python 3.11+
- UV package manager
- Google Cloud account (for Sheets API + optional storage)
- STT API keys (Azure, AssemblyAI, etc.)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd yi-voice-eval

# Install dependencies with UV
uv sync

# Copy config template
cp config/config.template.yaml config/config.yaml

# Edit config.yaml with your credentials and paths
```

### Credentials Setup

**Environment Variables (`.env` file):**

```bash
# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# STT API Keys
AZURE_SPEECH_KEY=your_key_here
AZURE_SPEECH_REGION=centralindia
ASSEMBLYAI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # If using Whisper API
```

**Google Sheets Access (User OAuth):**

1. Create Google Cloud project (owned by The Agency Fund)
2. Enable Google Sheets API + Google Drive API
3. Create OAuth 2.0 credentials (Desktop app type)
4. Download `credentials.json` → place in `/config` (gitignored)
5. First run will open browser for authentication → generates `token.json`
6. Share master results spreadsheet with team members beforehand

**For Colab:** Use Colab Secrets or runtime file upload for credentials (one-liner documented in notebooks)

### Running Notebooks

```bash
# Launch Jupyter with UV environment
uv run jupyter lab

# Notebooks will auto-detect config and credentials
```

### Config Validation

```bash
# Validate config before running 9-hour evaluation
uv run python -m yi_voice_eval.validate_config
```

## Pipeline Overview

### 1. Audio Preprocessing (`01_baseline_whisper.ipynb`)

**Input:** 41 raw audio files (mixed formats, sample rates, codecs)

**Processing:**

- Decode AMR-NB, AAC, MP3 codecs (via ffmpeg)
- Resample to 16kHz (standard STT input)
- Convert stereo → mono
- Normalize volume levels
- Export as WAV for consistency

**Output:**

- `/files/processed/` - Standardized audio files
- `/files/transcriptions/whisper_baseline/` - Whisper SRT transcriptions

**Rationale:** Preprocessing ensures fair comparison (all models receive identical input) while maintaining real-world audio characteristics (low bitrate, phone quality).

### 2. STT Model Comparison (`02_stt_comparison.ipynb`)

**Input:** Processed audio files + Whisper baseline

**Processing:**

- Run each STT model (Azure, AssemblyAI, Google) on all files
- Include retry logic (tenacity) for API rate limits/failures
- Log processing time, API costs, errors per file
- Convert all outputs to SRT format (standardized timestamps)

**Output:**

- `/files/transcriptions/{model_name}/` - SRT files per model
- `processing_logs.json` - Detailed execution logs

**Error Handling:** Failed files logged but don't block pipeline; retry once on transient errors.

### 3. Evaluation & Results (`03_results_analysis.ipynb`)

**Input:** All transcription SRTs + Whisper baseline (or ground truth when available)

**Processing:**

- Align transcriptions by timestamp
- Calculate WER, CER for each model vs baseline
- Compute custom WER for math-critical vocabulary
- Aggregate by file and overall
- Generate comparative statistics

**Output:**

- Google Sheets with two worksheets (see Results Format below)
- Timestamped worksheet appended to master spreadsheet
- Config snapshot included for reproducibility

## Evaluation Metrics

### Standard Metrics

**Word Error Rate (WER):**

- Measures word-level transcription accuracy
- Formula: `(Substitutions + Deletions + Insertions) / Total Words`
- Lower is better; production target: <15% for math tutoring use case

**Character Error Rate (CER):**

- Measures character-level accuracy (useful for morphologically rich languages like Kannada)
- More granular than WER, better captures partial correctness

### Custom Metrics (Future)

**Math Vocabulary WER:**

- Focused WER on numbers, mathematical terms, key tutoring phrases
- Rationale: Misrecognizing "2 + 3 = 5" is more critical than filler words
- Implementation: Weighted WER with importance list (last mile consideration)

### Concepts (One-liners)

- **WER:** Percentage of words transcribed incorrectly (industry standard for STT accuracy)
- **CER:** Percentage of characters transcribed incorrectly (better for complex scripts)
- **Sample Rate:** Audio frequency resolution (8kHz = phone quality, 16kHz = STT standard)
- **Codec:** Audio compression format (AMR-NB = adaptive low-bandwidth for voice calls)
- **Diarization:** Identifying and separating different speakers in audio
- **Ground Truth:** Human-verified correct transcription used for accuracy evaluation

## Results Format

### Google Sheets Structure

**Master Spreadsheet:** Owned by The Agency Fund, shared with Platform Commons and Youth Impact

**Worksheet 1: Aggregated Results** (`results_2025-01-15_1430`)

| Model            | Overall WER (%) | CER (%) | Cost/Hour ($) | Avg Latency (s) | Files Processed | Total Duration (hrs) | Notes                            |
| ---------------- | --------------- | ------- | ------------- | --------------- | --------------- | -------------------- | -------------------------------- |
| Whisper Large V3 | 12.3            | 8.1     | 0.00 (local)  | 2.3             | 41              | 9.01                 | Baseline (not ground truth)      |
| Azure Speech     | 18.7            | 11.2    | 9.18          | 1.1             | 41              | 9.01                 | kn-IN locale                     |
| AssemblyAI       | 15.4            | 9.6     | 8.10          | 1.8             | 40              | 8.92                 | 1 file failed (AMR decode issue) |
| Google STT V2    | -               | -       | -             | -               | -               | -                    | Pending validation               |

**Worksheet 2: Line-by-Line Results** (`details_2025-01-15_1430`)

| Audio File   | Timestamp | Reference (Whisper) | Azure           | AssemblyAI      | WER Azure (%) | WER AssemblyAI (%) | Audio URL                |
| ------------ | --------- | ------------------- | --------------- | --------------- | ------------- | ------------------ | ------------------------ |
| call_001.wav | 0:23-0:31 | ಬೀಜ ಸಮಾನ            | ಬಿಜ ಸಮಾನ        | ಬೀಜ ಸಮಾನಾ       | 12.5          | 5.0                | gs://bucket/call_001.wav |
| call_001.wav | 0:31-0:37 | ಎರಡು ಮತ್ತು ಮೂರು     | ಎರಡು ಮತ್ತು ಮುರು | ಎರಡು ಮತ್ತು ಮೂರು | 16.7          | 0.0                | gs://bucket/call_001.wav |

**Metadata Sheet:** Config snapshot, git commit, processing date, model versions

### Interpretation

- **Phase 1:** "Azure WER 18.7%" means "Azure differs from Whisper by 18.7%" (relative metric)
- **Phase 2:** "Azure WER 18.7%" means "Azure makes errors on 18.7% of words vs human transcription" (absolute metric)
- **Always check Notes column** to understand which ground truth was used

## Potential Failure Modes

### Colab-Specific Issues

- **Session timeouts:** 12hr limit (free), 90min if GPU unused → Solution: Break into batches or use Colab Pro
- **Storage limits:** 100GB temp storage → Solution: Stream from/to GCS directly with fsspec

### API Issues

- **Rate limiting:** STT APIs have quotas → Solution: Tenacity retry with exponential backoff
- **AMR-NB codec:** Not all libraries decode AMR-NB natively → Solution: Verify ffmpeg has amr codec support
- **Language code mismatches:** Some APIs use `kn-IN`, others `kan-IN` → Solution: Test per provider in config

### Processing Issues

- **Mixed sample rates:** Resampling 8kHz→16kHz may introduce artifacts → Solution: Test both raw and resampled
- **Stereo files:** 2% of files have separate channels (tutor/child?) → Solution: Investigate before simple mono conversion
- **Long files:** 32-minute files may exceed API limits → Solution: Chunk long files at silence points

### Multi-Organization Handoff

- **Credentials management:** Multiple orgs need access → Solution: Service accounts with least-privilege IAM roles
- **Colab vs local divergence:** Different runtime environments → Solution: UV + .env standardization, document differences

## Future Considerations

### Phase 2 Enhancements

**Diarization (Speaker Separation):**

- Use WhisperX (Whisper + Pyannote) to separate tutor vs child speech
- Evaluate STT on child segments only (more relevant to use case)
- Prerequisite: Professional transcriptions with speaker labels

**Noise Reduction Testing:**

- Compare STT performance on raw vs noise-reduced audio
- Test preprocessing strategies (spectral gating, Wiener filtering)
- Measure cost/benefit of preprocessing pipeline

**Fine-Tuning:**

- Fine-tune Whisper on Youth Impact dataset (requires ground truth)
- Evaluate custom model vs off-the-shelf options
- Decision point: Build vs buy analysis

### Advanced Metrics

**Code-Switching Analysis:**

- Measure performance on Kannada-English mixed speech
- Common in math tutoring (English number names, terminology)

**Age-Specific Performance:**

- If age labels available, analyze WER by child age group
- Younger children = different voice characteristics

**Error Type Analysis:**

- Categorize errors (substitution, insertion, deletion)
- Math-specific: Number transcription accuracy
- Tutoring-specific: Key phrase recognition

### Production Evolution

**API Service:**

- Wrap best-performing model in REST API
- Integrate with Youth Impact call platform
- Real-time transcription vs batch processing

**Active Learning Pipeline:**

- Continuously collect new voice samples
- Human-in-the-loop correction for edge cases
- Incremental model improvement

**Cost Optimization:**

- If fine-tuned model outperforms APIs, deploy self-hosted
- Evaluate on-device models for offline capability

## Contributing

This is an exploratory harness for The Agency Fund, Platform Commons, and Youth Impact collaboration.

**Primary Operator:** Platform Commons (software engineering)
**AI Evaluation Support:** The Agency Fund
**Domain Knowledge:** Youth Impact

For questions or issues, contact: [contact info]

## License

[TBD - discuss with stakeholders]
