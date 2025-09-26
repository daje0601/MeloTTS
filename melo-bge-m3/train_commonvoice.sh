#!/bin/bash
# Common Voice Korean 데이터로 MeloTTS BGE-M3 사전학습

echo "========================================"
echo "MeloTTS BGE-M3 Pretraining with Common Voice Korean"
echo "========================================"

# Step 1: 데이터 준비
echo "[1/3] Preparing Common Voice data..."
python prepare_commonvoice_ko.py --output-dir data/commonvoice_ko

# Step 2: Config 생성 확인
if [ ! -f "configs/config_commonvoice_pretrain.json" ]; then
    echo "❌ Config file not found!"
    exit 1
fi

# Step 3: 학습 시작
echo "[2/3] Starting pretraining..."
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

# 로그 디렉토리 생성
mkdir -p logs
mkdir -p checkpoints/commonvoice_pretrain

# 학습 실행
python -m melo.train \
    --config configs/config_commonvoice_pretrain.json \
    --model commonvoice_pretrain \
    2>&1 | tee logs/pretrain_commonvoice_$(date +%Y%m%d_%H%M%S).log

echo "[3/3] Pretraining completed!"
echo "Checkpoint saved in: checkpoints/commonvoice_pretrain/"