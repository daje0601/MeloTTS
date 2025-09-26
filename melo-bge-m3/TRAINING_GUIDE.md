# MeloTTS BGE-M3 학습 가이드

## 1. 데이터 준비

### 디렉토리 구조
```
melo-bge-m3/
├── data/
│   ├── wavs/                  # 음성 파일 (44100Hz WAV)
│   │   ├── speaker1_001.wav
│   │   ├── speaker1_002.wav
│   │   └── ...
│   ├── train_list.txt         # 학습 데이터 목록
│   └── val_list.txt           # 검증 데이터 목록
```

### 데이터 형식
`train_list.txt` 및 `val_list.txt` 형식:
```
wavs/speaker1_001.wav|안녕하세요. 오늘 날씨가 좋네요.|SPEAKER_001|KR
wavs/speaker1_002.wav|Hello, how are you today?|SPEAKER_001|EN
wavs/speaker2_001.wav|你好，今天天气很好。|SPEAKER_002|ZH
```

형식: `음성경로|텍스트|화자ID|언어코드`

- **언어코드**: KR, EN, ZH, JP (선택사항 - BGE-M3가 자동 감지)
- **화자ID**: 화자 식별자 (화자 클로닝용)

## 2. 환경 설정

```bash
# 가상환경 활성화
cd /service/ds/MeloTTS/melo-bge-m3
source ../.venv/bin/activate

# 필요한 패키지 설치 (uv 사용)
uv pip install -r requirements.txt
```

## 3. 학습 실행

### 단일 GPU 학습
```bash
./train.sh
```

### 다중 GPU 학습
```bash
# train.sh 수정
NUM_GPUS=4  # GPU 개수 변경
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 사용할 GPU 지정

./train.sh
```

### 직접 실행
```bash
# 단일 GPU
python -m melo.train --config configs/config_bge_m3.json --model melo_bge_m3

# 다중 GPU (4개)
torchrun --nproc_per_node=4 --master_port=29500 \
    -m melo.train --config configs/config_bge_m3.json --model melo_bge_m3
```

## 4. 설정 조정

`configs/config_bge_m3.json` 수정:

### 배치 크기
```json
"batch_size": 6,  // GPU 메모리에 따라 조정
```

### 학습률
```json
"learning_rate": 0.0003,  // 기본값
"lr_decay": 0.999875,      // 학습률 감소율
```

### Mixed Precision (메모리 절약)
```json
"fp16_run": true,  // FP16 사용 (속도/메모리 향상)
```

### 화자 수
```json
"n_speakers": 256,  // 최대 화자 수
```

## 5. 체크포인트

체크포인트는 `checkpoints/melo_bge_m3/` 디렉토리에 저장됩니다:
- `G_*.pth`: Generator (TTS 모델)
- `D_*.pth`: Discriminator
- `optimizer_*.pth`: Optimizer 상태

### 학습 재개
```bash
python -m melo.train \
    --config configs/config_bge_m3.json \
    --model melo_bge_m3 \
    --resume  # 마지막 체크포인트에서 재개
```

## 6. 모니터링

### TensorBoard
```bash
tensorboard --logdir logs/melo_bge_m3
```
브라우저에서 `http://localhost:6006` 접속

### 로그 확인
```bash
tail -f logs/train_melo_bge_m3_*.log
```

## 7. 주요 특징

### BGE-M3 통합
- 모든 언어 단일 BERT 모델 (1024차원)
- ja_bert 제거로 메모리 절약
- 향상된 토큰 커버리지 ([UNK] 최소화)

### 한국어 개선
- g2pkk 제거, jamo 직접 사용
- 더 빠른 전처리
- 간단한 의존성

### 다국어 지원
- 언어 태그 선택사항 (자동 감지)
- 혼합 언어 텍스트 지원
- 크로스링구얼 화자 클로닝

## 8. 문제 해결

### CUDA OOM
- `batch_size` 감소
- `segment_size` 감소 (16384 → 8192)
- `fp16_run: true` 확인

### 느린 학습
- `NUM_GPUS` 증가
- `batch_size` 증가 (메모리 허용 시)
- SSD에 데이터 저장

### 음질 문제
- 더 많은 epoch 학습 (최소 1000)
- 고품질 데이터 사용 (44.1kHz)
- `c_mel` 가중치 조정

## 9. 추론 테스트

학습 중 모델 테스트:
```python
from melo.api import TTS

# 모델 로드
tts = TTS(
    model_path="checkpoints/melo_bge_m3/G_latest.pth",
    config_path="configs/config_bge_m3.json",
    device="cuda"
)

# 음성 생성
tts.synthesize(
    "안녕하세요. BGE-M3로 학습한 모델입니다.",
    speaker_id=0,
    output_path="test.wav"
)
```