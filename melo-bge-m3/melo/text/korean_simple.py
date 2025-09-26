"""
Simplified Korean text processing using jamo directly
No g2pkk dependency - just pure jamo decomposition
"""

import re
from transformers import AutoTokenizer
from . import punctuation, symbols
from melo.text.ko_dictionary import english_dictionary, etc_dictionary
from jamo import h2j, j2hcj

# BGE-M3 model for Korean
model_id = 'BAAI/bge-m3'
tokenizer = AutoTokenizer.from_pretrained(model_id)

def normalize(text):
    """Normalize Korean text"""
    text = text.strip()
    # Remove Chinese characters
    text = re.sub("[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]", "", text)
    # Apply dictionary replacements
    text = normalize_with_dictionary(text, etc_dictionary)
    text = normalize_english(text)
    text = text.lower()
    return text

def normalize_with_dictionary(text, dic):
    """Replace text using dictionary"""
    if any(key in text for key in dic.keys()):
        pattern = re.compile("|".join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    return text

def normalize_english(text):
    """Convert English words to Korean pronunciation"""
    def fn(m):
        word = m.group()
        if word in english_dictionary:
            return english_dictionary.get(word)
        return word
    text = re.sub("([A-Za-z]+)", fn, text)
    return text

def korean_to_jamo(text):
    """
    Convert Korean text directly to jamo without g2pkk
    Simply decomposes Hangul into constituent jamo

    Example:
        "안녕" → "ㅇㅏㄴㄴㅕㅇ"
    """
    # Use jamo library directly
    jamo_text = j2hcj(h2j(text))
    return jamo_text

def text_normalize(text):
    """Normalize text for processing"""
    return normalize(text)

def distribute_phone(n_phone, n_word):
    """Distribute phones across words"""
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

def g2p(norm_text):
    """
    Simplified G2P using direct jamo decomposition
    No pronunciation rules, just character decomposition
    """
    tokenized = tokenizer.tokenize(norm_text)
    phs = []
    ph_groups = []

    for t in tokenized:
        # Handle BGE-M3's SentencePiece tokens
        if t.startswith("▁"):
            t = t[1:]  # Remove prefix
            if t:
                ph_groups.append([t])
        else:
            if ph_groups:
                ph_groups[-1].append(t)
            else:
                ph_groups.append([t])

    word2ph = []
    for group in ph_groups:
        text = "".join(group)

        # Skip empty strings
        if not text:
            continue

        # Handle punctuation
        if text in punctuation:
            phs += [text]
            word2ph += [1]
            continue

        # Convert to jamo (simple decomposition)
        # This doesn't apply pronunciation rules, just decomposes characters
        jamo_str = korean_to_jamo(text)

        # Each jamo character becomes a phoneme
        phonemes = list(jamo_str)

        # Filter out non-jamo characters (keep only Korean phonemes)
        phonemes = [p for p in phonemes if p in symbols or p in punctuation]

        if not phonemes:
            # If no valid phonemes, use silence
            phonemes = ['_']

        phone_len = len(phonemes)
        word_len = len(group)

        phone_distribution = distribute_phone(phone_len, word_len)
        word2ph += phone_distribution
        phs += phonemes

    # Add silence tokens
    phones = ["_"] + phs + ["_"]
    tones = [0 for _ in phones]  # Korean has no tones
    word2ph = [1] + word2ph + [1]

    # Ensure alignment with tokenized length
    expected_len = len(tokenized) + 2
    if len(word2ph) != expected_len:
        print(f"Warning: Alignment mismatch. word2ph={len(word2ph)}, expected={expected_len}")
        # Adjust to match
        if len(word2ph) < expected_len:
            word2ph += [1] * (expected_len - len(word2ph))
        else:
            word2ph = word2ph[:expected_len]

    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device='cuda'):
    """Get BERT features using unified BGE-M3"""
    from . import unified_bert
    return unified_bert.get_bert_feature(text, word2ph, device=device, model_id=model_id)

if __name__ == "__main__":
    # Test the simplified Korean processing
    test_texts = [
        "안녕하세요",
        "오늘 날씨가 좋네요",
        "AI 기술이 발전하고 있습니다",
        "😊 이모지도 처리됩니다",
    ]

    for text in test_texts:
        print(f"\nText: {text}")
        norm_text = text_normalize(text)
        print(f"Normalized: {norm_text}")

        # Direct jamo conversion (no g2pkk)
        jamo_text = korean_to_jamo(norm_text)
        print(f"Jamo: {jamo_text}")

        # Full G2P
        phones, tones, word2ph = g2p(norm_text)
        print(f"Phones: {phones}")
        print(f"Word2Ph: {word2ph}")