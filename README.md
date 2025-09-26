<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="300"/> <br>
  <a href="https://trendshift.io/repositories/8133" target="_blank"><img src="https://trendshift.io/api/badge/repositories/8133" alt="myshell-ai%2FMeloTTS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

## Introduction
MeloTTS is a **high-quality multi-lingual** text-to-speech library by [MIT](https://www.mit.edu/) and [MyShell.ai](https://myshell.ai). Supported languages include:

| Language | Example |
| --- | --- |
| English (American)    | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-US/speed_1.0/sent_000.wav) |
| English (British)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-BR/speed_1.0/sent_000.wav) |
| English (Indian)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN_INDIA/speed_1.0/sent_000.wav) |
| English (Australian)  | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-AU/speed_1.0/sent_000.wav) |
| English (Default)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-Default/speed_1.0/sent_000.wav) |
| Spanish               | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/es/ES/speed_1.0/sent_000.wav) |
| French                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/fr/FR/speed_1.0/sent_000.wav) |
| Chinese (mix EN)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/zh/ZH/speed_1.0/sent_008.wav) |
| Japanese              | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/jp/JP/speed_1.0/sent_000.wav) |
| Korean                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/kr/KR/speed_1.0/sent_000.wav) |

Some other features include:
- The Chinese speaker supports `mixed Chinese and English`.
- Fast enough for `CPU real-time inference`.

## Usage
- [Use without Installation](docs/quick_use.md)
- [Install and Use Locally](docs/install.md)
- [Training on Custom Dataset](docs/training.md)

The Python API and model cards can be found in [this repo](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#python-api) or on [HuggingFace](https://huggingface.co/myshell-ai).

**Contributing**

If you find this work useful, please consider contributing to this repo.

- Many thanks to [@fakerybakery](https://github.com/fakerybakery) for adding the Web UI and CLI part.

## Authors

- [Wenliang Zhao](https://wl-zhao.github.io) at Tsinghua University
- [Xumin Yu](https://yuxumin.github.io) at Tsinghua University
- [Zengyi Qin](https://www.qinzy.tech) (project lead) at MIT and MyShell

**Citation**
```
@software{zhao2024melo,
  author={Zhao, Wenliang and Yu, Xumin and Qin, Zengyi},
  title = {MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech},
  url = {https://github.com/myshell-ai/MeloTTS},
  year = {2023}
}
```

## License

This library is under MIT License, which means it is free for both commercial and non-commercial use.

## Korean Language Processing Notes

### Jamo Decomposition Alternative
The current implementation uses the `jamo` library for Korean character decomposition. This can be replaced with `hangul-utils` for the same functionality:

```python
# Current implementation (jamo)
from jamo import j2hcj, h2j
jamo_str = j2hcj(h2j(text))

# Alternative implementation (hangul-utils)
from hangul_utils import split_syllables
jamo_str = split_syllables(text, divide=False)
```

Note: When replacing, ensure the output format matches the symbol mapping in `melo/text/symbols.py`.

## Acknowledgements

This implementation is based on [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), [VITS2](https://github.com/daniilrobnikov/vits2) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). We appreciate their awesome work.


네, MeloTTS 코드베이스를 전체적으로 파악했습니다. 특히 요청하신 한국어 관련 부분을 중점적으로 분석했습니다:

  한국어 처리 핵심 요소

  1. 전처리 (melo/text/korean.py)

  - G2P: g2pkk 라이브러리 사용
  - 자모 분해: "안녕" → ['ᄋ', 'ᅡ', 'ᆫ', 'ᄂ', 'ᅧ', 'ᆼ']
  - 영어 처리: 영한 발음 사전 (ko_dictionary.py)
  - 정규화: 한자 제거, 특수문자 처리

  2. 학습 (melo/train.py)

  - BERT: kykim/bert-kor-base (768차원)
  - 음소: 57개 한국어 음소 심볼
  - 톤: 단일 톤 (중국어와 달리 성조 없음)
  - 데이터: WAV + 텍스트 메타데이터 필요

  3. 인퍼런스 (melo/infer.py)

  - 파이프라인: 텍스트 → G2P → 자모 → BERT → 음성 합성
  - 문장 분리: 중국어 스타일 분리기 사용
  - 속도 제어: speed, sdp_ratio 파라미터
  - 사전 학습 모델: myshell-ai/MeloTTS-Korean

  주요 특징:
  - 한영 혼용 텍스트 지원
  - CPU 실시간 추론 가능
  - VITS 기반 아키텍처 + BERT 컨텍스트화

> 꼭 저렇게 자모를 분리해야하는거야?

● 네, MeloTTS에서 자모 분리는 필수입니다. 이유를 설명드리겠습니다:

  자모 분리가 필요한 이유

  1. 음소 단위 모델링

  TTS 모델은 음소(phoneme) 단위로 학습합니다. 한글 음절 "안"은 하나의 단위가 아니라:
  - ᄋ (초성)
  - ᅡ (중성)
  - ᆫ (종성)

  이 3개의 독립적인 소리로 구성됩니다.

  2. 음성학적 정확성

  # melo/text/korean.py (line 70)
  jamo_str = j2hcj(h2j(g2pk))  # 음절 → 자모 변환
  - "학교" → "학꾜" (g2pkk) → ['ᄒ', 'ᅡ', 'ᆨ', 'ᄁ', 'ᅭ']
  - 음운 변화(경음화)를 자모 레벨에서 정확히 표현

  3. 모델 효율성

  - 음절 단위: 11,172개의 가능한 한글 음절
  - 자모 단위: 약 57개 심볼로 표현 가능
  - 학습 파라미터 대폭 감소

  4. 발음 변화 처리

  # 연음, 경음화, 격음화 등
  "먹는다" → "멍는다" → ['ᄆ', 'ᅥ', 'ᆼ', 'ᄂ', 'ᅳ', 'ᆫ', 'ᄃ', 'ᅡ']

  5. VITS 아키텍처 요구사항

  VITS 모델은 선형 음소 시퀀스를 입력으로 받아야 합니다. 음절은 2차원 구조(초성+중성+종성)라 직접 처리 불가능합니다.

  결론: 자모 분리는 한국어 TTS의 표준 접근법이며, 음성학적 정확성과 모델 효율성을 위해 필수적입니다.

> 파이썬에 jamo나 hangul-utils로 j2hcj를 대체할 수있나?

● 네, 가능합니다! 두 라이브러리 모두 자모 분해 기능을 제공합니다:

  라이브러리 비교

  1. jamo (현재 MeloTTS 사용 중)

  from jamo import j2hcj, h2j
  # "안녕" → "ㅇㅏㄴㄴㅕㅇ"
  result = j2hcj(h2j("안녕"))

  2. hangul-utils 대체 가능

  from hangul_utils import split_syllables
  # "안녕" → "ㅇㅏㄴㄴㅕㅇ" 
  result = split_syllables("안녕", divide=False)

  3. jamotools (더 강력한 대안)

  from jamotools import split_syllables
  # "안녕" → ['ㅇ', 'ㅏ', 'ㄴ', 'ㄴ', 'ㅕ', 'ㅇ']
  result = split_syllables("안녕")

  MeloTTS에서 대체하려면

  melo/text/korean.py (line 70) 수정:
  # 기존 코드
  from jamo import j2hcj, h2j
  jamo_str = j2hcj(h2j(g2pk))

  # hangul-utils로 대체
  from hangul_utils import split_syllables
  jamo_str = split_syllables(g2pk, divide=False)

  주의사항

  1. 출력 형식 차이
    - jamo: 호환 자모 (ㅇㅏㄴ)
    - hangul-utils: 호환 자모 옵션 있음
    - MeloTTS는 호환 자모를 요구함
  2. symbols.py 확인 필요
    - 자모 심볼 매핑이 일치해야 함
    - 초성/중성/종성 구분 필요
  3. 성능
    - jamo: 가볍고 빠름
    - hangul-utils: 더 많은 기능, 약간 느림

  결론: hangul-utils나 jamotools로 대체 가능하지만, 출력 형식을 MeloTTS의 심볼 체계와 맞춰야 합니다.

---