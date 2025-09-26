#!/usr/bin/env python3
"""
MeloTTS BGE-M3 데이터 준비 스크립트
음성 파일과 텍스트를 학습 형식으로 변환
"""

import os
import json
import wave
import contextlib
from pathlib import Path
from typing import List, Tuple
import argparse

def check_wav_format(wav_path: str) -> Tuple[bool, str]:
    """WAV 파일 형식 확인"""
    try:
        with contextlib.closing(wave.open(wav_path, 'r')) as f:
            channels = f.getnchannels()
            sample_width = f.getsampwidth()
            framerate = f.getframerate()
            frames = f.getnframes()
            duration = frames / float(framerate)

            issues = []
            if framerate != 44100:
                issues.append(f"샘플레이트 {framerate}Hz (44100Hz 필요)")
            if channels != 1:
                issues.append(f"채널 {channels} (모노 필요)")
            if sample_width != 2:
                issues.append(f"비트 깊이 {sample_width*8}bit (16bit 필요)")
            if duration > 15:
                issues.append(f"길이 {duration:.1f}초 (15초 이하 권장)")
            elif duration < 0.5:
                issues.append(f"너무 짧음 {duration:.1f}초")

            if issues:
                return False, ", ".join(issues)
            return True, f"OK ({duration:.1f}초)"
    except Exception as e:
        return False, str(e)

def prepare_from_folder(
    audio_dir: str,
    transcript_file: str = None,
    output_dir: str = "data",
    val_ratio: float = 0.1
):
    """
    폴더 구조에서 데이터 준비

    Args:
        audio_dir: 음성 파일 디렉토리
        transcript_file: 전사 파일 (없으면 파일명 사용)
        output_dir: 출력 디렉토리
        val_ratio: 검증 데이터 비율
    """

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)

    # 음성 파일 찾기
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(Path(audio_dir).glob(ext))

    print(f"찾은 음성 파일: {len(audio_files)}개")

    # 전사 파일 읽기 (있는 경우)
    transcripts = {}
    if transcript_file and os.path.exists(transcript_file):
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    filename = Path(parts[0]).stem
                    text = parts[1]
                    transcripts[filename] = text
        print(f"전사 파일 로드: {len(transcripts)}개")

    # 데이터 리스트 생성
    data_list = []

    for audio_path in audio_files:
        filename = audio_path.stem

        # 텍스트 찾기
        if filename in transcripts:
            text = transcripts[filename]
        else:
            # 전사가 없으면 파일명을 텍스트로 사용 (임시)
            text = filename.replace('_', ' ').replace('-', ' ')
            print(f"⚠️ 전사 없음: {filename} -> '{text}'")

        # 화자 ID 추출 (파일명 규칙에 따라)
        # 예: speaker001_text_001.wav -> speaker001
        if '_' in filename:
            speaker_id = filename.split('_')[0]
        else:
            speaker_id = "default"

        # 언어는 자동 감지되므로 비워둠 (또는 명시)
        lang = ""  # BGE-M3가 자동 감지

        # WAV 형식 확인
        if audio_path.suffix.lower() == '.wav':
            is_valid, msg = check_wav_format(str(audio_path))
            if not is_valid:
                print(f"❌ {filename}: {msg}")
                continue

        # 상대 경로로 변경
        rel_path = os.path.relpath(audio_path, output_dir)
        data_list.append(f"{rel_path}|{text}|{speaker_id}|{lang}")

    # 학습/검증 분할
    split_idx = int(len(data_list) * (1 - val_ratio))
    train_list = data_list[:split_idx]
    val_list = data_list[split_idx:]

    # 파일 저장
    with open(os.path.join(output_dir, "train_list.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_list))

    with open(os.path.join(output_dir, "val_list.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_list))

    print(f"\n✅ 데이터 준비 완료!")
    print(f"  학습 데이터: {len(train_list)}개")
    print(f"  검증 데이터: {len(val_list)}개")
    print(f"  출력 위치: {output_dir}/")

def prepare_kss_dataset():
    """KSS 데이터셋 준비 (한국어 단일 화자)"""
    print("KSS 데이터셋 준비...")

    # KSS 구조
    # kss/
    # ├── 1/
    # │   ├── 1_0000.wav
    # │   └── transcript.txt
    # └── 2/
    #     ├── 2_0000.wav
    #     └── transcript.txt

    kss_dir = "kss"
    if not os.path.exists(kss_dir):
        print(f"❌ KSS 디렉토리를 찾을 수 없습니다: {kss_dir}")
        print("다운로드: https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset")
        return

    data_list = []

    for folder in Path(kss_dir).iterdir():
        if not folder.is_dir():
            continue

        transcript_file = folder / "transcript.txt"
        if not transcript_file.exists():
            continue

        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    wav_name = parts[0]
                    text = parts[1]

                    wav_path = folder / wav_name
                    if wav_path.exists():
                        rel_path = str(wav_path)
                        # KSS는 한국어 단일 화자
                        data_list.append(f"{rel_path}|{text}|KSS|KR")

    # 분할 및 저장
    split_idx = int(len(data_list) * 0.9)

    os.makedirs("data", exist_ok=True)

    with open("data/train_list.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(data_list[:split_idx]))

    with open("data/val_list.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(data_list[split_idx:]))

    print(f"✅ KSS 데이터 준비 완료: {len(data_list)}개 샘플")

def create_sample_data():
    """테스트용 샘플 데이터 생성"""
    print("샘플 데이터 생성...")

    os.makedirs("data/wavs", exist_ok=True)

    # 샘플 텍스트
    samples = [
        ("sample_001.wav", "안녕하세요. 반갑습니다.", "speaker1", "KR"),
        ("sample_002.wav", "오늘 날씨가 참 좋네요.", "speaker1", "KR"),
        ("sample_003.wav", "인공지능 기술이 발전하고 있습니다.", "speaker1", "KR"),
        ("sample_004.wav", "Hello, how are you?", "speaker2", "EN"),
        ("sample_005.wav", "Nice to meet you.", "speaker2", "EN"),
    ]

    train_list = []
    val_list = []

    for i, (wav_name, text, speaker, lang) in enumerate(samples):
        wav_path = f"data/wavs/{wav_name}"

        # 실제 WAV 파일이 없으므로 경로만 생성
        line = f"{wav_path}|{text}|{speaker}|{lang}"

        if i < len(samples) * 0.8:
            train_list.append(line)
        else:
            val_list.append(line)

    with open("data/train_list.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_list))

    with open("data/val_list.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_list))

    print(f"✅ 샘플 데이터 생성 완료")
    print(f"⚠️ 실제 WAV 파일을 data/wavs/에 추가해야 합니다")

def main():
    parser = argparse.ArgumentParser(description='MeloTTS BGE-M3 데이터 준비')
    parser.add_argument('--mode', choices=['folder', 'kss', 'sample'],
                       default='sample',
                       help='데이터 준비 모드')
    parser.add_argument('--audio_dir', type=str,
                       help='음성 파일 디렉토리 (folder 모드)')
    parser.add_argument('--transcript', type=str,
                       help='전사 파일 경로 (folder 모드)')
    parser.add_argument('--output', type=str, default='data',
                       help='출력 디렉토리')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='검증 데이터 비율')

    args = parser.parse_args()

    if args.mode == 'folder':
        if not args.audio_dir:
            print("❌ --audio_dir 필요")
            return
        prepare_from_folder(args.audio_dir, args.transcript,
                          args.output, args.val_ratio)
    elif args.mode == 'kss':
        prepare_kss_dataset()
    elif args.mode == 'sample':
        create_sample_data()

if __name__ == "__main__":
    main()