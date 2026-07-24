# adcg — 광고 카피/이미지 자동 생성 파이프라인

소상공인 상품 사진과 매장 정보를 입력하면 광고 배경 이미지, 카피, 해시태그를
자동으로 생성하는 파이프라인입니다.

## 폴더 구조

```
adcg/
├── config.py                  # CLI 인자 파싱, product_info.json 생성, TONE_OPTIONS
├── pipeline.py                 # run_pipeline() — 전체 6단계 오케스트레이션
├── prompting/                  # 텍스트 생성 (GPU 불필요, OpenAI API만 사용)
│   ├── generator.py            #   - 씬(배경) 프롬프트 생성 (이미지 입력 필요)
│   ├── copywrite_generate.py   #   - generate_ad_copy(): 카피 생성
│   ├── hashtag_generate.py     #   - generate_ad_hashtags(): 해시태그 추천
│   ├── schema.py / system_prompt.py / encoding.py
├── preprocessing/               # 상품 배경 제거, 크롭, 마스크 (rembg)
├── generation/                  # ControlNet 조건부 확산 생성 (GPU 필요)
├── refinement/                  # 경계/디테일 복원
├── image_utils/                 # 블렌딩, 마스크 유틸
├── eval/                        # 정량 평가 (선택, --evaluate)
└── test_data/                   # 샘플 이미지/JSON

examples/
├── sample_product_info.json            # product_info.json 예시
├── quickstart_copy_and_hashtags.py     # 카피+해시태그만 (GPU 불필요)
└── quickstart_full_pipeline.py         # 전체 파이프라인 (GPU 필요)

requirements.txt        # 전체 파이프라인 (이미지 생성 포함, GPU)
requirements-text.txt   # 카피/해시태그만 (가볍게 테스트할 때)
requirements-eval.txt   # --evaluate 옵션의 hps_v2_score 지표용 (선택)
.env.example            # OPENAI_API_KEY 설정 예시
```

## 설치

카피/해시태그 기능만 빠르게 써보려면:

```bash
pip install -r requirements-text.txt
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY=sk-... 값을 채워넣으세요.
```

이미지 생성까지 포함한 전체 파이프라인을 쓰려면 (CUDA GPU 필요):

```bash
pip install -r requirements.txt
# torch는 CUDA 버전에 맞춰 https://pytorch.org/get-started/locally/ 안내에 따라
# 별도 설치가 필요할 수 있습니다.
```

## 1) 카피 + 해시태그만 생성해보기 (가장 빠른 경로)

```bash
python examples/quickstart_copy_and_hashtags.py \
    --info examples/sample_product_info.json \
    --tone 감성적인 \
    --copy-length 보통
```

코드로 직접 쓰는 경우:

```python
from adcg.config import TONE_OPTIONS
from adcg.prompting import generate_ad_copy, generate_ad_hashtags

product_info = {
    "product_name": "수제 버터 크루아상",
    "store_name": "아침빵집",
    "store_type": "동네 베이커리",
    "product_category": "베이커리",
    "product_description": "매일 아침 직접 반죽해 굽는 버터 크루아상",
    "features": ["프랑스산 버터 사용", "매일 아침 소량 생산"],
    "target_customer": "출근길 직장인",
    "promo_target": "재택 대신 출근을 시작한 20-30대 사회초년생",   # 홍보 대상
    "promotion": "",
    "price": "3,800원",
    "tone": TONE_OPTIONS[0],   # "친근한" / "고급스러운" / "미니멀한" / ...
    "draft_copy": "오늘 아침에도 갓 구운 크루아상 나왔어요!",         # 홍보 문구 초안
}

copy = generate_ad_copy(
    product_info=product_info,
    background_prompt="",       # 실제 파이프라인에서는 씬 생성 결과가 들어감
    model="gpt-5.4-nano",
)

hashtags = generate_ad_hashtags(
    product_info=product_info,
    tone=product_info["tone"],
    copy_length="보통",          # "짧게" / "보통" / "긴"
    model="gpt-5.4-nano",
)

print(copy)
print(hashtags["hashtags"])
```

`copy_length`는 UI의 "글 길이"(보통/긴 등) 선택값과 그대로 매핑되며,
`prompting/hashtag_generate.py`의 `HASHTAG_LENGTH_PRESETS`에서 해시태그
개수와 태그당 글자 수 제한을 함께 정의합니다.

## 2) 전체 파이프라인 실행 (이미지까지 생성)

```bash
python examples/quickstart_full_pipeline.py
```

또는 CLI로 직접:

```bash
python -m adcg.config \
    --image adcg/test_data/washing_machine_stand_01.png \
    --product-name "세탁기 받침대" \
    --store-name "홈리빙마켓" \
    --tone 미니멀한 \
    --copy-length 보통 \
    --promo-target "신혼부부, 자취 시작하는 1인 가구" \
    --draft-copy "세탁기 밑에 놓기만 해도 층간소음이 줄어요"
```

`run_pipeline()`은 다음 순서로 동작합니다.

1. 상품 전처리 (배경 제거/크롭)
2. 씬(배경) 프롬프트 생성
3. 광고 카피 + 해시태그 생성 → `output_dir/02_prompt/ad_copy.json`,
   `output_dir/02_prompt/ad_hashtags.json`
4. 조건부 확산 이미지 생성
5. 코어 상품 복원
6. 경계/디테일 복원 → 최종 이미지
7. (선택) 정량 평가 `--evaluate`

반환되는 `PipelineResult`에는 `copy_json`, `hashtag_json`, `final_image` 등
각 산출물 경로가 담겨 있습니다.

## tone 옵션

```python
TONE_OPTIONS = (
    "친근한", "고급스러운", "미니멀한",
    "감성적인", "전문적인", "위트있는", "흥미유발",
)
```

`--tone` CLI 인자는 이 값들 중 하나만 허용하며, `product_info.json`의
`tone` 필드에도 이 문자열 그대로 넣으면 됩니다.

## 홍보 대상 / 홍보 문구 초안

`product_info.json`에 다음 두 필드를 추가로 넣을 수 있습니다.

- `promo_target` (`--promo-target`): 이번 광고 캠페인이 겨냥하는 대상.
  `target_customer`(평소 주 고객층)와는 별개로, 특정 프로모션이 노리는
  타겟을 지정하고 싶을 때 사용합니다. 해시태그 생성 시 `target_customer`보다
  우선적으로 반영됩니다.
- `draft_copy` (`--draft-copy`): 사장님이 직접 써본 홍보 문구 초안.
  값이 있으면 `generate_ad_copy()`가 이 초안의 핵심 메시지와 사실 관계는
  유지하되, 요청한 톤에 맞춰 다시 다듬어 씁니다(그대로 복사하지 않음).
  해시태그 생성에서도 참고 문맥으로만 쓰입니다.

두 필드 모두 값이 없으면(`""` 또는 생략) 그냥 무시되므로, 필수 입력은
아닙니다.

## Google Colab에서 실행하기

`colab/` 폴더에 Colab용 노트북 2개를 준비해뒀습니다.

- **`colab/adcg_colab_text_only.ipynb`** — 카피 + 해시태그 생성만.
  GPU 불필요, 필요한 코드가 노트북 안에 전부 들어있어 이 zip을 업로드하지
  않아도 바로 실행됩니다. Colab에서 파일을 열고 `런타임 → 모두 실행`만
  누르면, API 키 입력 후 폼(Form) 필드로 상품 정보를 채워 결과를 받습니다.
- **`colab/adcg_colab_full_pipeline.ipynb`** — 이미지 생성까지 포함한 전체
  파이프라인. `런타임 유형을 T4 GPU 이상으로 변경`한 뒤, 노트북 안내에 따라
  이 프로젝트의 `adcg_project.zip`을 업로드(또는 Drive 마운트)하고 실행하면
  `requirements.txt`를 설치하고 `run_pipeline()`을 그대로 호출합니다.

두 노트북 다 `.ipynb`를 Colab(colab.research.google.com)에 업로드하거나
Google Drive에 넣고 "Colab으로 열기"로 실행하면 됩니다.
