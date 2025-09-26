"""
Unified BERT module using BGE-M3 for all languages
Replaces language-specific BERT modules with a single multi-lingual model
"""

import torch
from transformers import AutoTokenizer, AutoModel
import sys

# Global cache for model and tokenizer
models = {}
tokenizers = {}

def get_bert_feature(text, word2ph, device=None, model_id='BAAI/bge-m3'):
    """
    Extract BERT features using BGE-M3 for any language

    Args:
        text: Input text in any language
        word2ph: Word to phoneme alignment
        device: Device to run model on
        model_id: Model identifier (default: BGE-M3)

    Returns:
        Phone-level BERT features (1024-dimensional)
    """
    global models, tokenizers

    # Device selection
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer (cached)
    if model_id not in models:
        print(f"Loading BGE-M3 model for unified multi-lingual support...")
        model = AutoModel.from_pretrained(model_id).to(device)
        models[model_id] = model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizers[model_id] = tokenizer
        print(f"Model loaded successfully. Hidden size: {model.config.hidden_size}")
    else:
        model = models[model_id]
        tokenizer = tokenizers[model_id]

    # Tokenize and get features
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to(device)

        # Get hidden states from BGE-M3
        outputs = model(**inputs)
        # BGE-M3 uses last_hidden_state directly
        hidden_states = outputs.last_hidden_state[0].cpu()  # [seq_len, 1024]

    # Ensure alignment between tokens and phonemes
    # BGE-M3 may have different tokenization, so we need to handle this carefully
    token_len = inputs["input_ids"].shape[-1]

    # If word2ph doesn't match token length, we need to adjust
    if token_len != len(word2ph):
        # Simple interpolation strategy - can be improved
        if token_len > len(word2ph):
            # More tokens than words - aggregate
            ratio = token_len / len(word2ph)
            new_word2ph = []
            for w2p in word2ph:
                new_word2ph.extend([w2p] * int(ratio))
            # Adjust for rounding
            while len(new_word2ph) < token_len:
                new_word2ph.append(word2ph[-1])
            while len(new_word2ph) > token_len:
                new_word2ph.pop()
            word2ph = new_word2ph[:token_len]
        else:
            # Fewer tokens than words - need to handle this case
            # This might happen with subword tokenization
            # For now, truncate word2ph (not ideal, should be fixed in preprocessing)
            word2ph = word2ph[:token_len]

    # Distribute features to phone level
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = hidden_states[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T  # [1024, total_phones]