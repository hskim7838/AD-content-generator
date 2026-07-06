# ad-copy-llm-pipeline

생성형 AI 기반 소상공인 광고 콘텐츠 서비스의 일부 — **여러 텍스트 LLM(GPT-5.5 / GPT-5.4 mini / GPT-5.4 nano)으로 광고 문구(카피라이팅)를 생성하고, 정량적으로 비교·평가**하는 모듈입니다.

## 파이프라인 상 위치

```
1) 누끼 이미지 → tiny.json (상품 속성 추출)      ← data/generate_tiny_jsons.py 는 이 결과물의 샘플
2) 생성 모델 비교 (이미지 생성 모델)              ← 이 레포의 범위 밖
3) Stable Diffusion inpainting 등 배경 이미지 생성 ← 이 레포의 범위 밖
4) 생성 방향 설정                                ← 이 레포의 범위 밖
5) OpenCV로 문구 합성                            ← 이 레포의 범위 밖 (text_composer.py, 별도 관리)

[이 레포]  tiny.json + 매장 정보 → 광고 카피 생성(여러 모델) → 정량 평가
```

## 레포 구조

```
ad-copy-llm-pipeline/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── compare_copy_llms.py      # 여러 LLM에 동일 입력으로 카피 생성 요청 + 비교 DataFrame 생성
├── eval_copy.py              # 생성 결과에 세 축(규칙기반/임베딩유사도/LLM judge) 평가 컬럼 추가
├── data/
│   └── generate_tiny_jsons.py   # 1단계 tiny.json 샘플 데이터 생성 스크립트
└── examples/
    └── run_comparison.py        # 전체 흐름(로드→생성→평가) 실행 예시 CLI 스크립트
```

## 설치

```bash
pip install -r requirements.txt
```

## API 키 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 실제 키를 채워넣으세요.

```
OPENAI_API_KEY=sk-...
```

## 사용법

### 1) tiny.json 샘플 데이터 준비 (선택)

이미 1단계에서 만든 tiny.json이 있다면 이 단계는 건너뛰어도 됩니다. 예시 데이터로 테스트해보고 싶다면:

```bash
python data/generate_tiny_jsons.py --output-dir ./data/tiny_jsons
```

`./data/tiny_jsons/` 아래에 상품 5종(베이커리/스시/꽃다발/김치/원피스)의 `.json` 파일이 생성됩니다.

### 2) 카피 생성 + 비교 + 평가 (CLI)

```bash
python examples/run_comparison.py \
    --tiny-json ./data/tiny_jsons/flower_shop_bouquet.json \
    --tone 미니멀 \
    --store-name "fluffy flower" \
    --store-phone "02-123-4567" \
    --store-price "12,000원"
```

- `--tone` : `미니멀` / `강조형` / `친근한` / `고급스러운` / `위트있는` 중 선택
- `--skip-semantic`, `--skip-judge` : 비용을 줄이고 규칙 기반 체크만 빠르게 보고 싶을 때 사용

### 3) 코드에서 직접 사용 (로컬/주피터랩)

```python
import json
from compare_copy_llms import compare_models
from eval_copy import evaluate_dataframe, show_full, print_comments

with open("./data/tiny_jsons/flower_shop_bouquet.json", encoding="utf-8") as f:
    product_attrs = json.load(f)

store_info = {"name": "fluffy flower", "phone": "02-123-4567", "price": "12,000원"}
tone = "미니멀"

base_df = compare_models(product_attrs, store_info, tone=tone)
result = evaluate_dataframe(base_df, product_attrs, store_info, tone=tone)

show_full(result)       # 표 전체를 잘리지 않게 표시
print_comments(result)  # judge_comment를 모델별로 줄글 출력
```

### 4) 코랩에서 사용

```python
!git clone https://github.com/<your-id>/ad-copy-llm-pipeline.git
!pip install -q -r ad-copy-llm-pipeline/requirements.txt

import sys
sys.path.append("/content/ad-copy-llm-pipeline")

import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")

from compare_copy_llms import compare_models
from eval_copy import evaluate_dataframe, show_full, print_comments
```

코드를 수정한 뒤에는 push 후 코랩에서 `!git -C /content/ad-copy-llm-pipeline pull`로 최신화하고, 이미 import했던 세션이라면 아래로 캐시를 지운 뒤 다시 import하세요.

```python
for name in ["eval_copy", "compare_copy_llms"]:
    sys.modules.pop(name, None)
```

## 비교 대상 모델 변경

`compare_copy_llms.py`의 `BACKENDS` 리스트만 수정하면 됩니다.

```python
BACKENDS: List[BackendConfig] = [
    BackendConfig(name="GPT-5.5", provider="openai_compatible", model="gpt-5.5"),
    BackendConfig(name="GPT-5.4 mini", provider="openai_compatible", model="gpt-5.4-mini"),
    BackendConfig(name="GPT-5.4 nano", provider="openai_compatible", model="gpt-5.4-nano"),
]
```

Naver CLOVA Studio(HyperCLOVA X)처럼 OpenAI 호환 엔드포인트를 제공하는 모델도 `base_url`만 지정하면 동일한 방식으로 추가할 수 있습니다.

## 평가 축 설명

`eval_copy.py`가 `compare_models()` 결과에 추가하는 컬럼입니다.

| 컬럼 | 무엇을 측정하는가 | 계산 방식 | 좋은 수치 기준 |
|---|---|---|---|
| `rule_pass` | 하드 제약(글자수/가격/금칙어) 준수 여부 | 정규식/문자열 비교 (API 호출 없음) | **True**가 정상. False면 사실 오류 가능성 |
| `rule_violations` | rule_pass=False일 때 구체적 위반 사유 | 위 체크에서 걸린 항목 나열 | 빈 문자열이 정상 |
| `faithfulness_score` | 생성 카피가 원본(tiny.json+매장정보)과 의미적으로 얼마나 가까운지 | 임베딩 코사인 유사도 (0~1) | 절대값보다 **모델 간 상대 비교**용. 지나치게 낮으면 할루시네이션 의심 신호 |
| `judge_tone_fit` | 요청 톤과의 일치도 | LLM judge 채점 (1~5) | 4~5점 양호 |
| `judge_persuasiveness` | 구매/방문 유도 설득력 | LLM judge 채점 (1~5) | 4~5점 양호 |
| `judge_naturalness` | 한국어 표현의 자연스러움 | LLM judge 채점 (1~5) | 4~5점 양호 |
| `judge_constraint_adherence` | 형식 제약(글자수 등) 준수 정도 | LLM judge 채점 (1~5) | 5점이 정상 |
| `judge_total` | 위 4개 항목 평균 | 산술 평균 | 4.0 이상 실사용 가능, 4.5 이상 우수 |

**판단 우선순위**: `rule_pass=True`를 먼저 전제 조건으로 걸러낸 뒤, 그 안에서 `judge_total`로 순위를 매기는 순서를 권장합니다. LLM judge는 "글로서 잘 썼다"는 인상 위주로 채점하기 때문에, 가격 오류나 금칙어가 있어도 점수를 후하게 줄 수 있습니다.

## 설정값 조정

`eval_copy.py` 상단에서 프로젝트 상황에 맞게 바꿀 수 있습니다.

- `MAX_TITLE_LEN`, `MAX_SUBTITLE_LEN` : 글자 수 제한
- `BANNED_WORDS` : 금칙어(과장 표현) 목록
- `JUDGE_MODEL` : LLM judge로 쓸 모델. 생성 후보군(GPT-5.x)과 겹치지 않는 모델(기본 `gpt-4o`)을 권장 — self-preference bias 방지

## 참고: 반복 실험에서 관찰된 패턴 (예시)

톤/상품 조합을 바꿔가며 여러 번 실행해본 결과, 아래와 같은 경향이 관찰되었습니다 (표본이 적어 참고용입니다 — 실제 결론을 내리려면 반복 실행 + 집계가 필요합니다).

- **GPT-5.4 mini**: 항상 가장 빠름(latency 최저). "친근한"/"고급스러운"처럼 무난한 표현이 유리한 톤에서 강세. 다만 매장명·구체 상품명을 생략하는 경향이 반복 관찰됨 (faithfulness_score가 상대적으로 낮게 나오는 원인).
- **GPT-5.4 nano**: 실제 상품 속성(메뉴명, 매장명 등)을 카피에 직접 반영하는 경향. "위트있는"/"미니멀"처럼 디테일 한두 개가 임팩트를 좌우하는 톤에서 강세. 다만 정보를 나열식으로 담아 자연스러움(naturalness) 점수가 깎이는 경우가 있음.
- **GPT-5.5**: latency가 가장 느리면서, 단독 1위를 기록한 사례는 상대적으로 적었음.
- **"고급스러운"** 같은 뉘앙스가 섬세한 톤은 세 모델 모두 tone_fit이 낮게 나오는 경향 — 시스템 프롬프트에 톤별 어휘 가이드나 few-shot 예시를 보강할 필요가 있어 보임.

여러 톤 × 여러 상품 조합을 자동 반복 실행하고 모델별/톤별 평균을 집계하는 기능은 추후 추가 예정입니다.
