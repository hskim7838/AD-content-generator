# 소상공인 광고 카피·배너 생성 파이프라인

상품 사진 + 매장 정보를 입력하면, 여러 GPT 모델·톤(tone) 조합으로 광고 카피를 생성하고,
정량 평가 후 가장 점수가 높은 카피를 상품 사진에 합성해 배너 이미지(PNG)로 만드는 파이프라인입니다.

## 전체 실행 순서

```
1) copywrite_step4_5.py            상품 속성 + 매장 정보 → GPT 호출 → 카피 문구 9개 생성
2) step6_evaluate_generated_copy.py 생성된 카피를 규칙 기반 + 임베딩 + LLM judge로 정량 평가
3) step7_visualize_eval_result.py   평가 결과를 막대그래프 PNG로 시각화
4) step8_compose_banner.py         평가 점수 1위 카피를 상품 사진에 합성해 배너 PNG 저장
```

각 단계는 서로 다른 파이썬 프로세스로 실행되며, 파이썬 변수를 공유하지 않고
**중간 결과 파일(JSON/CSV)을 통해서만** 데이터를 주고받습니다.

```
copywrite_step4_5.py  ──▶ generated_copy_results.json
step6_evaluate_...py  ──▶ generated_copy_eval_result.csv  (+ generated_copy_results.json 재참조)
step7_visualize_...py ──▶ eval_result_*.png
step8_compose_banner.py ──▶ generated_banners/*.png
```

---

## 1. 필요한 폴더/파일 구조

```
프로젝트_루트/
├── .env                          # ⚠ 직접 생성 필요 (아래 4번 항목 참고)
├── compare_copy_llms.py          # copywrite_step4_5.py 최초 실행 시 자동 생성됨
├── copywrite_step4_5.py
├── eval_copy.py                  # ⚠ 별도 보유 파일. step6가 import해서 사용
├── step6_evaluate_generated_copy.py
├── step7_visualize_eval_result.py
├── step8_compose_banner.py
│
├── tiny_dataset/
│   ├── jsons/                    # 상품 속성 (id.json 여러 개 가능)
│   │   └── latte_001.json
│   └── images/                   # 상품 사진
│       └── latte_001.jpg
│
├── option_pools.json             # 매장 정보 + 톤/브랜드/판매문구 옵션 풀
├── run_config.json               # 이번 실행에 tiny_dataset 중 어떤 상품을 쓸지 지정
│
├── fonts/                        # (선택) 한글 폰트를 직접 넣는 폴더
│
├── generated_copy_results.json   # [산출물] STEP1이 생성
├── generated_copy_eval_result.csv# [산출물] STEP6이 생성
└── generated_banners/            # [산출물] STEP8이 생성
    └── {product_id}_{tone}_{model}.png
```

### 1-1. `tiny_dataset/jsons/{product_id}.json` — 상품 속성

이미지 캡셔닝 등(구 STEP1~3)으로 만들어지는, **상품 사진 자체에서 추출한 정보**입니다.
필수 키는 없지만 `image`(사진 경로)와 `id`(없으면 파일명으로 대체)를 넣어두면
STEP1/STEP8이 이 값으로 상품을 식별합니다.

```json
{
  "id": "latte_001",
  "image": "./tiny_dataset/images/latte_001.jpg",
  "category": "음료",
  "name": "시그니처 카페라떼",
  "caption": "부드러운 우유 거품 위에 하트 라떼아트를 올린 따뜻한 카페라떼",
  "color": ["아이보리", "브라운"],
  "material": "세라믹 머그컵",
  "size": "355ml (Tall)",
  "keywords": ["라떼", "커피", "핸드드립"]
}
```

### 1-2. `option_pools.json` — 매장/마케팅 옵션 풀

상품 사진만으로는 알 수 없는 **매장 정보**와, 톤·모델 비교 실험을 위해
랜덤 조합을 뽑아낼 **옵션 풀**입니다. `store_info`, `tones`, `products`, `brands`, `sales`
5개 키가 반드시 있어야 합니다.

```json
{
  "store_info": {
    "store_name": "브루웍스 커피",
    "location": "서울 마포구 연남동",
    "hours": "매일 08:00 - 21:00",
    "signature": "핸드드립 & 시그니처 라떼",
    "phone": "02-1234-5678"
  },
  "tones": ["미니멀", "강조형", "친근한", "고급스러운", "위트있는"],
  "products": ["시그니처 카페라떼", "바닐라빈 라떼", "아인슈페너"],
  "brands": ["브루웍스 커피", "브루웍스 로스터리"],
  "sales": ["신메뉴 출시 기념 15% 할인", "2잔 구매 시 쿠키 증정", "평일 오전 1+1"]
}
```

> 실제 서비스에서는 이 값들이 랜덤 추첨이 아니라 **사용자가 직접 입력한 값**으로
> 대체될 자리입니다. 지금은 모델·톤별 정량 비교 실험을 위한 더미 옵션 풀입니다.

### 1-3. `run_config.json` — 이번 실행에 사용할 상품 지정

`copywrite_step4_5.py`는 한 번 실행에 상품 하나만 처리하므로,
`tiny_dataset/jsons/` 중 어떤 걸 쓸지 이 파일이 지정합니다.

```json
{
  "run_config_source_id": "latte_001",
  "product_attrs_path": "./tiny_dataset/jsons/latte_001.json"
}
```

> tiny_dataset 안의 상품을 전부 자동 순회하도록 `copywrite_step4_5.py`를 바꾸면
> 이 파일은 없앨 수 있습니다 (현재 코드 기준으로는 필요).

### 1-4. `eval_copy.py` — STEP6 정량 평가 모듈 (별도 보유)

`step6_evaluate_generated_copy.py`가 `from eval_copy import evaluate_dataframe, show_full`로
불러다 씁니다. 이 README가 관리하는 범위 밖의 파일이며, **STEP6·STEP7 실행 전 반드시
같은 폴더에 있어야** 합니다.

### 1-5. 한글 폰트 (STEP8)

STEP8은 배너에 한글 텍스트를 그리기 위해 TTF/OTF/TTC 폰트가 필요합니다. 아래 순서로 자동 탐색합니다.

1. `FONT_PATH` 환경변수
2. `./fonts/` 폴더 안의 `.ttf`/`.otf`/`.ttc` 파일
3. OS 기본 한글 폰트 (macOS: AppleGothic / Windows: 맑은 고딕 / Linux: Noto Sans CJK)

Linux에서 안 잡히면 `sudo apt install fonts-noto-cjk`로 설치하거나, `./fonts/`에
Noto Sans KR 등 한글 폰트 파일을 직접 넣으세요.

---

## 2. `.env` 설정 (직접 생성 필요)

이 README 배포본에는 `.env`가 포함되어 있지 않습니다. 프로젝트 루트에 아래 내용으로
`.env` 파일을 **직접 만들어야** `copywrite_step4_5.py`, `step6_evaluate_generated_copy.py`가
정상 실행됩니다.

```
OPENAI_API_KEY=sk-...여기에_실제_키...
```

- `python-dotenv`가 각 스크립트 시작 시 `load_dotenv()`로 이 파일을 자동으로 읽습니다.
- 키가 없으면 `RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다...")`로 즉시 실행이 중단됩니다.
- `.env`는 git 등 버전관리에 올리지 마세요 (`.gitignore`에 추가 권장).

---

## 3. 실행 방법

```bash
# 0) 최초 1회: 패키지 설치
pip install pandas openai pydantic python-dotenv pillow

# 1) 카피 문구 생성 (모델 3종 × 조합 9개)
python copywrite_step4_5.py
#  -> generated_copy_results.json 생성

# 2) 정량 평가 (규칙 기반 + 임베딩 유사도 + LLM judge)
python step6_evaluate_generated_copy.py
#  -> generated_copy_eval_result.csv 생성

# 3) 평가 결과 시각화
python step7_visualize_eval_result.py
#  -> 막대그래프 PNG 저장

# 4) 배너 이미지 합성
python step8_compose_banner.py
#  -> generated_banners/{product_id}_{tone}_{model}.png 저장
```

각 스크립트는 이전 단계의 산출물이 없으면 어떤 파일이 없어서 실행할 수 없는지
`FileNotFoundError` 메시지로 알려줍니다.

---

## 4. 산출물 요약

| 파일/폴더 | 생성 단계 | 내용 |
|---|---|---|
| `compare_copy_llms.py` | STEP1 (최초 1회) | GPT 호출용 모듈, 코드 안에 소스가 내장돼 있다가 파일로 저장됨 |
| `generated_copy_results.json` | STEP1 | 상품별 원본 정보(`products`) + 생성된 카피 9개(`results`) |
| `generated_copy_eval_result.csv` | STEP6 | 카피별 정량 평가 점수 (`judge_total` 등) |
| `eval_result_*.png` | STEP7 | 모델/톤별 평균 점수 막대그래프 |
| `generated_banners/*.png` | STEP8 | 최종 배너 이미지 (기본: 상품당 1위 카피만 합성) |

---

## 5. 자주 발생하는 문제

- **`OPENAI_API_KEY가 설정되지 않았습니다`** → `.env` 파일이 실행 위치 기준 올바른 경로에 있는지, 키 오탈자가 없는지 확인
- **`한글을 지원하는 폰트를 찾지 못했습니다` (STEP8)** → `./fonts/`에 한글 폰트 파일을 넣거나 `FONT_PATH` 환경변수 지정
- **STEP6에서 `ModuleNotFoundError: eval_copy`** → `eval_copy.py`가 `step6_evaluate_generated_copy.py`와 같은 폴더에 있는지 확인
- **STEP8에서 이미지 없음 경고** → `tiny_dataset/jsons/*.json`의 `image` 경로가 실제 파일 위치와 일치하는지 확인
