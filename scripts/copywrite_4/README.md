# 광고 카피라이팅 생성 파이프라인 (STEP 1~5, VSCode/로컬 실행용)

`copywritee_step1_5.py` 하나로 "상품 이미지 → 자동 캡션 → 옵션 조합 → GPT 카피 생성"까지
전 과정을 실행하는 로컬 파이프라인입니다. 이미지 합성(배너 제작)은 포함하지 않고,
카피 문구(title/subtitle/price/cta) 생성까지만 다룹니다.

## 전체 흐름

```
STEP 1. 환경 설정 (OpenAI 클라이언트, 매장 정보)
STEP 2. 이미지 폴더 -> tiny.json 자동 생성 (GPT-4o Vision으로 캡션 자동 생성)
STEP 3. tiny.json 통합 -> option_pools.json / run_config.json 자동 생성
STEP 4. compare_copy_llms.py 모듈 파일 생성 (카피 생성 함수)
STEP 5. 랜덤 조합 9개 -> GPT 실제 호출 -> 카피 문구(title/subtitle/price/cta) 출력
```

STEP1~5는 전부 `copywritee_step1_5.py` **한 파일 안에서 순서대로** 실행됩니다.
(정량 평가·시각화를 하는 STEP6/STEP7은 이 문서에서 다루지 않습니다.)

## 사전 준비

1. 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```
   `ModuleNotFoundError: No module named 'dotenv'` 에러가 나면 이 단계가 안 된 것이니,
   `pip install python-dotenv`로 따로 설치해도 됩니다. (가상환경을 쓴다면 활성화 상태인지도 확인)

2. `.env` 파일 생성 후 API 키 입력
   ```
   OPENAI_API_KEY=sk-...
   ```
   `.env`는 이름이 점(`.`)으로 시작하는 숨김 파일이라 파일 탐색기에 안 보일 수 있습니다.
   터미널에서 `ls -la`로 존재 여부를 확인하세요.

3. 상품 이미지 준비 (아래 "INPUT_DIR 설정" 참고)

## INPUT_DIR 설정 (상품 이미지 경로)

STEP2가 `INPUT_DIR`에 지정된 이미지(또는 이미지가 든 폴더)를 스캔해서 GPT-4o Vision으로
캡션을 자동 생성합니다. 특정 사용자 이름이 박힌 절대경로를 코드에 하드코딩하지 않도록,
아래 3가지 방법 중 하나로 지정할 수 있게 되어 있습니다 (우선순위 순서).

1. **커맨드라인 인자로 직접 지정** (가장 우선)
   ```bash
   python3 copywritee_step1_5.py /본인/이미지/경로/bakery.png
   ```
2. **`.env` 또는 환경변수**
   ```
   INPUT_DIR=/본인/이미지/경로/bakery.png
   ```
3. **기본값** — 위 둘 다 안 주면 스크립트와 같은 폴더의 `./bakery.png`를 찾습니다.

`INPUT_DIR`은 이미지 파일 하나를 직접 가리켜도 되고, 여러 이미지가 든 폴더를 가리켜도 됩니다
(폴더면 안에 있는 `.png/.jpg/.jpeg/.webp` 전부를 스캔합니다).

## 실행

```bash
python3 copywritee_step1_5.py
```

정상적으로 끝나면 아래가 생성됩니다.

```
tiny_dataset/
  images/                 원본 이미지 복사본
  jsons/                  이미지별 tiny.json (id, image, new_caption, store_info)
option_pools.json          매장/톤/상품 옵션 풀
run_config.json             이번 실행에 쓰인 옵션
compare_copy_llms.py        카피 생성 모듈 (STEP4가 자동으로 새로 씀 — 있어도 덮어쓰여지므로 신경 안 써도 됨)
```

콘솔에는 조합 9개 각각의 `title / subtitle / price / cta`가 출력됩니다.

## 사용 모델

`BACKENDS_POOL`에 등록된 3개 모델을 라운드로빈으로 순환 배정해서 조합마다 하나씩 사용합니다.

```python
BACKENDS_POOL = [
    BackendConfig(name="GPT-5.5", provider="openai_compatible", model="gpt-5.5"),
    BackendConfig(name="GPT-5.4 mini", provider="openai_compatible", model="gpt-5.4-mini"),
    BackendConfig(name="GPT-5.4 nano", provider="openai_compatible", model="gpt-5.4-nano"),
]
```

`gpt-4.5-mini` / `gpt-4.5-nano`는 실제로 존재하지 않는 모델명이라 API 에러가 나고, 해당 조합은
결과에서 통째로 빠지게 됩니다(에러 난 조합은 STEP5가 자동으로 건너뛰도록 되어 있음).
반드시 실제 존재하는 모델명(`gpt-5.4-mini` / `gpt-5.4-nano` 등)을 사용하세요.

## 자주 겪는 에러

| 에러 메시지 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: No module named 'dotenv'` | `python-dotenv` 미설치 | `pip install -r requirements.txt` |
| `OPENAI_API_KEY가 설정되지 않았습니다` | `.env` 파일이 없거나 키가 비어있음 | `.env`에 `OPENAI_API_KEY=sk-...` 추가 |
| `tiny_dataset/jsons 안에 tiny.json이 없습니다` | STEP2가 `INPUT_DIR`을 못 찾아서 캡션 생성을 건너뜀 | 위 "INPUT_DIR 설정" 참고, 콘솔 로그에서 `⚠ 입력 폴더/파일을 찾을 수 없습니다` 또는 `❌ 캡션 자동 생성 실패` 로그 확인 |
| 특정 모델(GPT-5.5 등) 결과만 나오고 나머지는 빔 | `BACKENDS_POOL`에 존재하지 않는 모델명이 섞여있음 | 실제 존재하는 모델명으로 수정 |

## 참고

- `tiny_dataset/`, `option_pools.json`, `run_config.json`, `compare_copy_llms.py`는 전부
  실행할 때마다 자동 생성/갱신되는 산출물이라, 미리 준비해둘 필요는 없습니다.
- `STORE_INFO`는 이미지 하나하나가 아니라 **전체 이미지에 공통 적용**되는 매장 정보입니다.
  이미지마다 다른 매장 정보를 쓰고 싶다면 STEP2 루프 안에서 이미지별로 분기하도록 수정이 필요합니다.
