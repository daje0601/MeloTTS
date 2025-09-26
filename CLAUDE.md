# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

MeloTTS is a multi-lingual text-to-speech (TTS) library using VITS-based architecture with BERT contextualization. It supports 10 languages including Korean, Chinese, Japanese, English variants, Spanish, and French.

## Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# For Korean TTS specifically, ensure these are installed:
pip install g2pkk>=0.1.1 jamo==0.4.1 transformers==4.27.4
```

### Running Inference
```bash
# Web UI (Gradio interface)
python melo/app.py

# API Server
python melo/api.py

# Command-line inference
python melo/main.py
```

### Training
```bash
# Train model (requires config file)
python melo/train.py -c configs/config.json -m model_name

# Resume training from checkpoint
python melo/train.py -c configs/config.json -m model_name -r
```

## Architecture Overview

### Core Components

1. **Text Processing Pipeline** (`melo/text/`)
   - Language-specific modules: `korean.py`, `chinese.py`, `japanese.py`, `english.py`, etc.
   - Each language has: normalize(), text_to_sequence(), g2p conversion
   - Korean uses g2pkk for G2P and Jamo decomposition

2. **Model Architecture** (`melo/models.py`)
   - VITS-based synthesizer with BERT integration
   - Language embeddings for multi-lingual support
   - Duration predictor, flow-based decoder, HiFi-GAN vocoder

3. **Training Pipeline** (`melo/train.py`)
   - Distributed training support
   - Bucket sampling for efficient batching
   - Mixed precision training with automatic scaling

4. **Inference Engine** (`melo/infer.py`)
   - TTS class for high-level inference
   - Sentence splitting with language-specific rules
   - Speed and noise scale controls for prosody

## Korean Language Processing Details

### Text Preprocessing Flow
1. **Normalization** (`melo/text/korean.py:normalize()`)
   - Removes Chinese characters
   - Dictionary-based replacements (`ko_dictionary.py`)
   - English word to Korean phonetic mapping

2. **G2P Conversion** (`melo/text/korean.py:latin_to_hangul()`)
   - Uses g2pkk library for Korean G2P
   - Jamo decomposition: syllables → individual consonants/vowels
   - Example: "안녕" → ['ᄋ', 'ᅡ', 'ᆫ', 'ᄂ', 'ᅧ', 'ᆼ']

3. **BERT Encoding**
   - Model: `kykim/bert-kor-base` (768-dim features)
   - Phone-to-word alignment using `distribute_phone()`

### Korean-Specific Symbols
- 57 Korean phoneme symbols defined in `melo/text/symbols.py`
- Includes all Hangul Jamo characters
- Single tone level (no tonal variations)

### Pre-trained Models
- Available at HuggingFace: `myshell-ai/MeloTTS-Korean`
- Checkpoint URLs in `melo/utils.py:DOWNLOAD_CKPT_URLS`

## Data Format for Training

### Required Structure
```
data/
├── wavs/           # Audio files (WAV format)
├── metadata.csv    # Text annotations
└── speakers.json   # Speaker ID mapping
```

### Metadata Format
```
audio_path|text|speaker_id|language
```

### Korean Text Processing Features
- Mixed script support (Hangul + English)
- Automatic English romanization
- Dictionary-based normalization for common terms
- Sentence splitting using Chinese-style splitter (not Latin)

## Key Dependencies for Korean

- **g2pkk**: Korean G2P conversion (essential)
- **jamo**: Hangul Jamo decomposition
- **transformers**: Korean BERT model loading
- **anyascii**: Optional ASCII romanization

## Performance Optimization

- Global G2P model caching to avoid reload
- Batch processing for multiple sentences
- GPU acceleration (CUDA/MPS support)
- Streaming support for long texts
- CPU real-time inference capable

## Important Configuration

### Language IDs
- Korean: 'KR' (ID: 4)
- Chinese: 'ZH' (ID: 0)
- English: 'EN' (ID: 1)
- Japanese: 'JP' (ID: 3)

### Model Hyperparameters
- Sample rate: 44100 Hz
- Hop size: 512
- Filter length: 2048
- BERT hidden size: 768 (Korean/Japanese), 1024 (Chinese)

## Debugging Korean TTS Issues

1. **G2P Failures**: Check g2pkk installation and model loading
2. **BERT Errors**: Ensure transformers==4.27.4 (specific version required)
3. **Audio Quality**: Adjust speed (0.8-1.2) and sdp_ratio (0.0-1.0)
4. **Mixed Text**: Enable English romanization if needed