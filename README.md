# NSMC 감성분석 — 스타터 코드

CAS2250 인공지능개론 학기 프로젝트용입니다. 여러분이 채울 곳은 **두 군데**입니다.

| 파일 | 뭐가 들어 있나 |
|---|---|
| `nsmc_data.py` | 데이터 다운로드, 로딩, 라벨 잡음 감사 — **그대로 쓰세요** |
| `tokenize_util.py` | 토크나이저와 Dataset — 그대로 (토크나이저는 백본과 세트입니다) |
| `metrics.py` | 정확도·F1·혼동행렬 — 그대로 (팀마다 다르게 재면 비교가 안 됩니다) |
| `baseline.py` | TF-IDF + 로지스틱 회귀 — 그대로, 먼저 돌려서 기준선을 잡으세요 |
| **`train.py`** | **TODO 1 — `training_step`** |
| **`predict.py`** | **TODO 2 — `predict_loop`** |
| `evaluate.py` | 정확도·혼동행렬을 `results.json` 으로 — 그대로 |
| `error_analysis.py` | 틀린 예측 추출과 분류표 — 그대로 |

## 시작하기

```bash
pip install -r requirements.txt
python nsmc_data.py            # 데이터 내려받기 + 라벨 잡음 확인
python baseline.py             # 기준선. CPU로 1분이면 끝납니다
```

`nsmc_data.py` 의 출력은 그냥 넘기지 마세요. 이 데이터의 상한이 왜 100%가
아닌지가 거기 숫자로 나옵니다. 보고서에 쓰실 내용입니다.

## 파인튜닝

```bash
python train.py    --model klue/roberta-base --seed 0
python train.py    --model klue/roberta-base --seed 1
python train.py    --model klue/roberta-base --seed 2
python evaluate.py --ckpt runs/klue_roberta-base_seed0 \
                          runs/klue_roberta-base_seed1 \
                          runs/klue_roberta-base_seed2
```

GPU 권장입니다. Colab 무료 T4에서 2 epoch에 20~40분 정도 걸립니다.
CPU로도 돌긴 하지만 몇 시간 걸립니다. 먼저 `--limit-train 4000` 으로
코드가 도는지만 확인하세요.

고를 수 있는 백본은 강의 12번 슬라이드에 있습니다.
`klue/roberta-base`, `beomi/KcELECTRA-base`, `monologg/koelectra-base-v3-discriminator`.
어느 걸 왜 골랐는지가 보고서에 들어가야 합니다.

## 에러 분석

```bash
python error_analysis.py --ckpt runs/klue_roberta-base_seed0
# errors_top100.csv 를 열어서 category 칼럼을 손으로 채웁니다
python error_analysis.py --summarize runs/klue_roberta-base_seed0/errors_top100.csv
```

이 단계는 자동화할 수 없습니다. 100건을 직접 읽으셔야 합니다.
읽다 보면 '이건 모델 잘못이 아닌데' 싶은 게 나옵니다. 그게 핵심입니다.

## 막혔을 때

| 증상 | 십중팔구 원인 |
|---|---|
| `NotImplementedError` | TODO 를 아직 안 채우셨습니다. docstring 에 단계가 적혀 있습니다 |
| 정확도가 50% 근처 | 라벨이 섞였거나 lr 이 너무 큽니다. 2e-5 부터 시작하세요 |
| 정확도가 베이스라인보다 낮음 | 토크나이저와 모델 이름이 다른지 확인하세요. 조용히 나빠집니다 |
| CUDA out of memory | `--bs 16` 으로 줄이세요 |
| 2 epoch 넘기니 나빠짐 | 정상입니다. 과적합이에요. 강의 16번 슬라이드 |
| 시드마다 결과가 널뜀 | lr 을 낮추세요. 정상 범위는 ±0.1%p 안쪽입니다 |
| `baseline.py` 가 87%인데 내 모델이 85% | 아직 안 끝난 겁니다. 91% 근처가 나와야 정상입니다 |

## 규칙

- **NSMC 로 이미 파인튜닝된 체크포인트를 받아다 쓰는 것은 안 됩니다.**
  사전학습 백본(`klue/roberta-base` 등)을 불러오는 건 당연히 괜찮습니다.
  그게 이 과제의 핵심이에요.
- `Trainer` 를 써서 `training_step` 을 우회하는 것도 안 됩니다.
  `training_step` 과 `predict_loop` 이 채점 대상입니다.
- `metrics.py` 는 고치지 마세요. 모두가 같은 방식으로 재야 비교가 됩니다.
- 시드는 최소 세 개 돌리고, 평균과 표준편차를 같이 보고하세요.
- **테스트셋으로 튜닝하지 마세요.** 5만 건 test 를 보면서 설정을 고르면
  그 숫자는 이미 오염된 겁니다. train 에서 validation 을 따로 떼세요.

## 제출물

제출 파일 하나로 압축해서 내세요. 압축 파일 이름은 `<학번>_task1_nsmc.zip` 입니다.

```
<학번>_task1_nsmc/
├── report.pdf                        보고서 1~2쪽
├── code/                             스타터 폴더 전체 (TODO 를 채운 것)
│   ├── train.py                      <- 여러분이 채운 파일
│   ├── predict.py                    <- 여러분이 채운 파일
│   └── (나머지 8개는 받은 그대로)
└── results/
    ├── baseline.json                 baseline.py 출력
    ├── results.json                  evaluate.py 출력 (시드 3개)
    ├── errors_top100_*.csv           category 칼럼을 손으로 채운 것
    ├── label_fixes.csv               수정한 라벨 목록
    └── *_train.json                  train.py 가 남긴 학습 기록
```

`label_fixes.csv` 는 직접 만드세요. 칼럼은 `id, document, old_label, new_label, reason` 입니다.
고친 라벨이 없으면 빈 파일 대신 보고서에 "왜 안 고쳤는지" 를 적으세요.

### 넣지 마세요

| 빼야 할 것 | 이유 |
|---|---|
| `runs/<모델>_seed*/` 체크포인트 폴더 | 용량이 매우 큽니다. 시드 3개면 더 커지고요 |
| `data/` | NSMC 원본입니다. 코드가 알아서 받습니다 |
| `__pycache__/`, `.venv/` | |

체크포인트는 채점에 쓰지 않습니다. `results.json` 과 코드로 재현합니다.
그래서 **README 에 적은 명령이 그대로 돌아가는 것** 이 중요합니다.

## 제출 전 체크

- [ ] 위 '제출물' 구조대로 묶었고, 체크포인트와 `data/` 를 뺐는가
- [ ] `pip install -r requirements.txt` 만으로 환경이 재현되는가
- [ ] README 에 적은 명령을 그대로 복사해서 붙이면 도는가
- [ ] 보고서의 모든 숫자가 그 명령들의 출력에서 나왔는가
- [ ] 베이스라인부터 최종까지의 결과 표가 있는가 (시드 3회 평균 ± 표준편차)
- [ ] 혼동행렬이 들어 있는가
- [ ] 틀린 예측 100건 분류표와 예시가 들어 있는가
- [ ] 수정한 라벨 목록과 전후 정확도가 있는가
- [ ] 시도했다가 실패한 것도 적었는가
