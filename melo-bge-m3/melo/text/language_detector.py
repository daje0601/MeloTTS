"""
Unicode-based language detection for automatic language identification
"""

import re
from typing import List, Tuple, Dict

def detect_language_by_unicode(text: str) -> str:
    """
    Detect language using Unicode ranges

    Args:
        text: Input text

    Returns:
        Language code (ZH, KR, JP, EN, etc.)
    """
    # Count characters by language
    lang_counts = {
        'ZH': 0,  # Chinese
        'KR': 0,  # Korean
        'JP': 0,  # Japanese
        'EN': 0,  # English/Latin
        'RU': 0,  # Russian
        'AR': 0,  # Arabic
    }

    for char in text:
        code_point = ord(char)

        # Korean Hangul
        if 0xAC00 <= code_point <= 0xD7AF or \
           0x1100 <= code_point <= 0x11FF or \
           0x3130 <= code_point <= 0x318F:
            lang_counts['KR'] += 1

        # Chinese (CJK Unified Ideographs)
        elif 0x4E00 <= code_point <= 0x9FFF or \
             0x3400 <= code_point <= 0x4DBF or \
             0x20000 <= code_point <= 0x2A6DF:
            lang_counts['ZH'] += 1

        # Japanese Hiragana and Katakana
        elif 0x3040 <= code_point <= 0x309F or \
             0x30A0 <= code_point <= 0x30FF or \
             0xFF66 <= code_point <= 0xFF9F:
            lang_counts['JP'] += 1

        # Latin alphabet (English and European languages)
        elif 0x0041 <= code_point <= 0x005A or \
             0x0061 <= code_point <= 0x007A or \
             0x00C0 <= code_point <= 0x00FF:
            lang_counts['EN'] += 1

        # Cyrillic (Russian)
        elif 0x0400 <= code_point <= 0x04FF:
            lang_counts['RU'] += 1

        # Arabic
        elif 0x0600 <= code_point <= 0x06FF or \
             0x0750 <= code_point <= 0x077F:
            lang_counts['AR'] += 1

    # Handle special cases
    # Japanese text often contains Kanji (Chinese characters)
    if lang_counts['JP'] > 0 and lang_counts['ZH'] > 0:
        # If we have Hiragana/Katakana, it's likely Japanese
        if lang_counts['JP'] > lang_counts['ZH'] * 0.3:
            return 'JP'

    # Chinese mixed with English
    if lang_counts['ZH'] > 0 and lang_counts['EN'] > 0:
        if lang_counts['ZH'] > lang_counts['EN']:
            return 'ZH_MIX_EN'

    # Return language with most characters
    if sum(lang_counts.values()) == 0:
        return 'EN'  # Default to English

    return max(lang_counts, key=lang_counts.get)


def split_mixed_language(text: str) -> List[Tuple[str, str]]:
    """
    Split text by language boundaries for mixed-language text

    Args:
        text: Input text possibly containing multiple languages

    Returns:
        List of (text_segment, language_code) tuples
    """
    segments = []
    current_segment = []
    current_lang = None

    for char in text:
        char_lang = detect_language_by_unicode(char)

        # Skip whitespace and punctuation for language detection
        if char.isspace() or char in '.,!?;:()[]{}"""\'':
            if current_segment:
                current_segment.append(char)
            continue

        if char_lang != current_lang:
            # Language boundary detected
            if current_segment:
                segment_text = ''.join(current_segment).strip()
                if segment_text:
                    segments.append((segment_text, current_lang))
            current_segment = [char]
            current_lang = char_lang
        else:
            current_segment.append(char)

    # Add final segment
    if current_segment:
        segment_text = ''.join(current_segment).strip()
        if segment_text:
            segments.append((segment_text, current_lang))

    return segments


def get_language_g2p_func(language: str):
    """
    Get the appropriate G2P function for a language

    Args:
        language: Language code

    Returns:
        G2P function for the language
    """
    g2p_map = {
        'ZH': 'chinese',
        'ZH_MIX_EN': 'chinese',
        'EN': 'english',
        'JP': 'japanese',
        'KR': 'korean',
        'ES': 'spanish',
        'FR': 'french',
        'RU': 'english',  # Fallback to English for unsupported
        'AR': 'english',  # Fallback to English for unsupported
    }

    module_name = g2p_map.get(language, 'english')

    # Dynamically import the appropriate module
    if module_name == 'chinese':
        from . import chinese
        return chinese.g2p
    elif module_name == 'korean':
        from . import korean
        return korean.g2p
    elif module_name == 'japanese':
        from . import japanese
        return japanese.g2p
    elif module_name == 'english':
        from . import english
        return english.g2p
    elif module_name == 'spanish':
        from . import spanish
        return spanish.g2p
    elif module_name == 'french':
        from . import french
        return french.g2p
    else:
        from . import english
        return english.g2p


def process_mixed_language_text(text: str) -> Tuple[List, List, List]:
    """
    Process mixed-language text and return phones, tones, and word2ph

    Args:
        text: Input text possibly containing multiple languages

    Returns:
        Tuple of (phones, tones, word2ph)
    """
    segments = split_mixed_language(text)

    all_phones = ['_']  # Start token
    all_tones = [0]
    all_word2ph = [1]

    for segment_text, language in segments:
        g2p_func = get_language_g2p_func(language)

        # Get phones, tones, word2ph for this segment
        phones, tones, word2ph = g2p_func(segment_text)

        # Remove start/end tokens from segment (they're already added)
        if phones[0] == '_':
            phones = phones[1:]
            tones = tones[1:]
            word2ph = word2ph[1:]
        if phones[-1] == '_':
            phones = phones[:-1]
            tones = tones[:-1]
            word2ph = word2ph[:-1]

        # Adjust tone indices based on language
        # This maintains compatibility with existing tone embedding
        tone_offset = get_tone_offset(language)
        adjusted_tones = [t + tone_offset if t > 0 else 0 for t in tones]

        all_phones.extend(phones)
        all_tones.extend(adjusted_tones)
        all_word2ph.extend(word2ph)

    all_phones.append('_')  # End token
    all_tones.append(0)
    all_word2ph.append(1)

    return all_phones, all_tones, all_word2ph


def get_tone_offset(language: str) -> int:
    """
    Get tone offset for a language to maintain compatibility

    Args:
        language: Language code

    Returns:
        Tone offset value
    """
    # From symbols.py
    tone_offsets = {
        'ZH': 0,
        'ZH_MIX_EN': 0,
        'JP': 6,   # After Chinese tones
        'EN': 7,   # After Japanese
        'KR': 11,  # After English tones
        'ES': 12,  # After Korean
        'FR': 13,  # After Spanish
        'RU': 14,  # After French
        'AR': 15,  # After Russian
    }

    return tone_offsets.get(language, 0)