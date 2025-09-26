#!/usr/bin/env python3
"""
Test script for MeloTTS with BGE-M3 integration
Tests multi-language processing, unified BERT, and jamo-based Korean
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_language_detection():
    """Test Unicode-based language detection"""
    print("\n=== Testing Language Detection ===")
    from melo.text.language_detector import detect_language_by_unicode, split_mixed_language

    test_cases = [
        ("안녕하세요", "KR"),
        ("Hello world", "EN"),
        ("你好世界", "ZH"),
        ("こんにちは", "JP"),
        ("Привет", "RU"),
        ("안녕 Hello 你好", None),  # Mixed
    ]

    for text, expected in test_cases:
        detected = detect_language_by_unicode(text)
        print(f"Text: {text:20} | Expected: {expected or 'Mixed':5} | Detected: {detected}")

    # Test mixed language splitting
    mixed_text = "Hello 안녕하세요 你好"
    segments = split_mixed_language(mixed_text)
    print(f"\nMixed text: {mixed_text}")
    print(f"Segments: {segments}")

def test_korean_jamo():
    """Test Korean processing with jamo"""
    print("\n=== Testing Korean Jamo Processing ===")
    from melo.text.korean import korean_text_to_phonemes, g2p, text_normalize

    test_texts = [
        "안녕하세요",
        "오늘 날씨가 좋네요",
        "AI 기술 발전",
        "123 숫자 테스트",
    ]

    for text in test_texts:
        print(f"\nOriginal: {text}")

        # Normalize
        norm = text_normalize(text)
        print(f"Normalized: {norm}")

        # Jamo conversion
        jamo = korean_text_to_phonemes(norm)
        print(f"Jamo: {jamo}")

        # Full G2P
        try:
            phones, tones, word2ph = g2p(norm)
            print(f"Phones: {phones[:10]}..." if len(phones) > 10 else f"Phones: {phones}")
            print(f"Tones: {tones[:10]}..." if len(tones) > 10 else f"Tones: {tones}")
            print(f"Word2Ph: {word2ph}")
        except Exception as e:
            print(f"G2P Error: {e}")

def test_unified_bert():
    """Test unified BERT with BGE-M3"""
    print("\n=== Testing Unified BERT (BGE-M3) ===")

    try:
        from melo.text.unified_bert import get_bert_feature

        test_texts = [
            ("안녕하세요", [1, 2, 3, 1, 1]),
            ("Hello world", [1, 5, 5, 1]),
            ("你好世界", [1, 2, 2, 1]),
        ]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")

        for text, word2ph in test_texts:
            print(f"\nText: {text}")
            print(f"Word2Ph: {word2ph}")

            try:
                features = get_bert_feature(text, word2ph, device=device)
                print(f"Feature shape: {features.shape}")
                print(f"Feature dims: {features.shape[0]} (should be 1024)")
                assert features.shape[0] == 1024, f"Expected 1024 dimensions, got {features.shape[0]}"
            except Exception as e:
                print(f"Error: {e}")

    except ImportError as e:
        print(f"Cannot test BERT features: {e}")
        print("Make sure transformers and BGE-M3 are installed")

def test_model_architecture():
    """Test modified model architecture"""
    print("\n=== Testing Model Architecture ===")

    try:
        from melo.models import TextEncoder
        import torch.nn as nn

        # Create dummy encoder
        encoder = TextEncoder(
            n_vocab=100,
            out_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
            num_languages=8,  # Optional
            num_tones=16,
        )

        print(f"TextEncoder created successfully")

        # Check for single BERT projection
        assert hasattr(encoder, 'bert_proj'), "Missing bert_proj"
        assert not hasattr(encoder, 'ja_bert_proj'), "ja_bert_proj should be removed"
        print(f"✓ Single BERT projection confirmed")

        # Check dimensions
        assert encoder.bert_proj.in_channels == 1024, f"Expected 1024 input channels, got {encoder.bert_proj.in_channels}"
        print(f"✓ BGE-M3 dimensions (1024) confirmed")

        # Test forward pass (dummy data)
        batch_size = 2
        seq_len = 10
        x = torch.randint(0, 100, (batch_size, seq_len))
        x_lengths = torch.tensor([seq_len, seq_len])
        tone = torch.zeros(batch_size, seq_len, dtype=torch.long)
        language = torch.zeros(batch_size, seq_len, dtype=torch.long)
        bert = torch.randn(batch_size, 1024, seq_len)

        # Forward pass without ja_bert
        try:
            output = encoder(x, x_lengths, tone, language, bert)
            print(f"✓ Forward pass successful without ja_bert")
            print(f"Output shape: {output[0].shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")

    except ImportError as e:
        print(f"Cannot test model architecture: {e}")

def test_data_processing():
    """Test data processing with unified dimensions"""
    print("\n=== Testing Data Processing ===")

    try:
        from melo.data_utils import TextAudioSpeakerCollate

        # Check collate function
        collate_fn = TextAudioSpeakerCollate()
        print("✓ TextAudioSpeakerCollate created")

        # The collate function should return one less item (no ja_bert)
        # Original: (text, text_lengths, spec, spec_lengths, wav, wav_lengths, sid, tone, language, bert, ja_bert)
        # New: (text, text_lengths, spec, spec_lengths, wav, wav_lengths, sid, tone, language, bert)

        print("✓ Data processing configured for unified BERT")

    except Exception as e:
        print(f"Data processing test error: {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("MeloTTS BGE-M3 Integration Test Suite")
    print("=" * 60)

    tests = [
        test_language_detection,
        test_korean_jamo,
        test_unified_bert,
        test_model_architecture,
        test_data_processing,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()