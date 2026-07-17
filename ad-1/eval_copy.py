# -*- coding: utf-8 -*-
"""
eval_copy.py
-------------
compare_copy_llms.py 의 compare_models() 결과 DataFrame에
"세 축 평가" 컬럼을 추가하는 모듈.

  1) 규칙 기반 체크 (rule_pass, rule_violations)
     - 글자 수 제한, 가격/연락처 등 하드 팩트 일치 여부, 금칙어 포함 여부
     - API 호출 없음(비용 0), 가장 먼저/빠르게 걸러내는 용도
     - rule_pass=False 인 경우 반드시 원인을 확인해야 함
       (사실 오류, 예: 가격 불일치는 실사용 불가)

  2) 시맨틱 유사도 (faithfulness_score)
     - 임베딩(OpenAI text-embedding-3-small)으로 "생성된 카피"와
       "원본 tiny.json + 매장 정보"의 의미적 근접도(0~1)를 측정
     - 참조 정답이 없는 임베딩 유사도이므로 절대적인 "합격선"은 없고,
       모델 간 상대 비교용으로 사용. 다만 지나치게 낮으면(예: 0.2 이하)
       원본과 무관한 내용을 지어냈을 가능성을 의심해볼 신호

  3) LLM judge (judge_tone_fit / judge_persuasiveness / judge_naturalness /
     judge_constraint_adherence / judge_total)
     - 사용자가 정의한 rubric(톤 일치도/설득력/자연스러움/제약 준수)에 따라
       채점 전용 모델(기본 gpt-4o, 생성 후보군과 겹치지 않는 모델 권장)로 1~5점 채점
     - judge_total = 위 4개 항목의 평균

사용법
------
    from compare_copy_llms import compare_models
    from eval_copy import evaluate_dataframe

    df = compare_models(product_attrs, store_info, tone="친근한")
    result = evaluate_dataframe(df, product_attrs, store_info, tone="친근한")
    result

사전 준비
--------
    pip install openai pandas numpy python-dotenv pydantic
    OPENAI_API_KEY 환경변수(또는 .env 파일)가 설정되어 있어야 함
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

# ----------------------------------------------------------------------------
# 0. 설정값 (프로젝트 상황에 맞게 조정)
# ----------------------------------------------------------------------------
MAX_TITLE_LEN = 25          # 타이틀 최대 글자 수
MAX_SUBTITLE_LEN = 35       # 서브카피 최대 글자 수
BANNED_WORDS = ["최고", "1위", "업계 최고", "완벽", "무조건", "유일", "보장", "100%"]

EMBEDDING_MODEL = "text-embedding-3-small"
JUDGE_MODEL = "gpt-4o"      # 생성 후보군(GPT-5.x 등)과 겹치지 않는 모델을 채점자로 권장


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)


# ----------------------------------------------------------------------------
# 1. 규칙 기반 체크 (rule_pass / rule_violations)
# ----------------------------------------------------------------------------
def rule_based_check(row: pd.Series, store_info: dict) -> Tuple[bool, List[str]]:
    """
    row        : compare_models() 결과의 한 행 (title/subtitle/price/cta 포함)
    store_info : 원본 매장 정보 (가격 등 하드 팩트 대조용)
    return     : (통과 여부, 위반 사유 리스트)
    """
    violations: List[str] = []

    title = row.get("title") or ""
    subtitle = row.get("subtitle") or ""
    price = row.get("price") or ""
    full_text = f"{title} {subtitle} {price} {row.get('cta') or ''}"

    # 1) 글자 수 제한
    if len(title) > MAX_TITLE_LEN:
        violations.append(f"타이틀 {len(title)}자 (제한 {MAX_TITLE_LEN}자 초과)")
    if len(subtitle) > MAX_SUBTITLE_LEN:
        violations.append(f"서브카피 {len(subtitle)}자 (제한 {MAX_SUBTITLE_LEN}자 초과)")

    # 2) 가격 등 하드 팩트 일치 여부 (숫자만 뽑아 비교 -> "12,000원" vs "12000" 등 표기 차이 허용)
    store_price = store_info.get("price")
    if store_price:
        store_digits = re.sub(r"\D", "", str(store_price))
        copy_digits = re.sub(r"\D", "", price)
        if store_digits and copy_digits and store_digits != copy_digits:
            violations.append(f"가격 불일치 (원본 {store_price} vs 생성 {price})")

    # 3) 금칙어(과장 표현) 체크
    hit_words = [w for w in BANNED_WORDS if w in full_text]
    if hit_words:
        violations.append(f"금칙어 포함: {', '.join(hit_words)}")

    return (len(violations) == 0), violations


# ----------------------------------------------------------------------------
# 2. 시맨틱 유사도 (faithfulness_score)
# ----------------------------------------------------------------------------
def _get_embedding(text: str) -> np.ndarray:
    client = _get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(response.data[0].embedding)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def faithfulness_score(row: pd.Series, product_attrs: dict, store_info: dict) -> Optional[float]:
    """
    생성된 카피와 원본 정보(tiny.json + 매장 정보) 간 임베딩 코사인 유사도(0~1).
    낮을수록 원본과 무관한 내용(할루시네이션 가능성)을 만들었을 확률이 높다는 신호.
    """
    source_text = json.dumps(product_attrs, ensure_ascii=False) + " " + \
                  json.dumps(store_info, ensure_ascii=False)
    copy_text = f"{row.get('title') or ''} {row.get('subtitle') or ''} " \
                f"{row.get('price') or ''} {row.get('cta') or ''}"

    if not copy_text.strip():
        return None

    source_emb = _get_embedding(source_text)
    copy_emb = _get_embedding(copy_text)
    return round(_cosine_similarity(source_emb, copy_emb), 4)


# ----------------------------------------------------------------------------
# 3. LLM judge (judge_tone_fit / persuasiveness / naturalness / constraint_adherence / total)
# ----------------------------------------------------------------------------
class JudgeScore(BaseModel):
    tone_fit: int = Field(ge=1, le=5, description="요청한 톤앤매너와 얼마나 일치하는가 (1~5)")
    persuasiveness: int = Field(ge=1, le=5, description="구매/방문을 유도하는 설득력 (1~5)")
    naturalness: int = Field(ge=1, le=5, description="한국어 문장으로서 자연스러운 정도 (1~5)")
    constraint_adherence: int = Field(ge=1, le=5, description="글자수 등 형식 제약 준수 정도 (1~5)")
    comment: str = Field(description="채점 근거를 1~2문장으로 간단히")


JUDGE_SYSTEM_PROMPT = """\
당신은 광고 카피라이팅 품질을 채점하는 엄격한 평가자입니다.
아래 4개 기준 각각을 1~5점으로 채점하세요 (5점 = 매우 우수).
- tone_fit: 요청된 톤앤매너와의 일치도
- persuasiveness: 구매/방문을 유도하는 설득력
- naturalness: 한국어 표현의 자연스러움 (번역투, 어색한 조사 등이 없는지)
- constraint_adherence: 타이틀 12자 내외, 서브카피 20자 내외 등 형식 제약 준수 정도

반드시 아래 JSON 스키마로만 답하세요.
{schema}
"""


def llm_judge_score(
    row: pd.Series,
    product_attrs: dict,
    store_info: dict,
    tone: str,
    judge_model: str = JUDGE_MODEL,
) -> JudgeScore:
    client = _get_client()
    schema_hint = json.dumps(JudgeScore.model_json_schema(), ensure_ascii=False, indent=2)
    system = JUDGE_SYSTEM_PROMPT.format(schema=schema_hint)
    user = f"""\
[상품 속성]
{json.dumps(product_attrs, ensure_ascii=False, indent=2)}

[매장 정보]
{json.dumps(store_info, ensure_ascii=False, indent=2)}

[요청했던 톤]
{tone}

[채점 대상 카피]
- title: {row.get('title')}
- subtitle: {row.get('subtitle')}
- price: {row.get('price')}
- cta: {row.get('cta')}
"""
    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    return JudgeScore.model_validate(json.loads(raw))


# ----------------------------------------------------------------------------
# 4. 종합: DataFrame에 세 축 컬럼 추가
# ----------------------------------------------------------------------------
def evaluate_dataframe(
    df: pd.DataFrame,
    product_attrs: dict,
    store_info: dict,
    tone: str,
    judge_model: str = JUDGE_MODEL,
    run_semantic: bool = True,
    run_judge: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    compare_models() 결과 df에 세 축 평가 컬럼을 추가해서 반환한다.
    error가 있던 행(모델 호출 자체가 실패한 행)은 평가를 건너뛰고 컬럼을 비워둔다.

    추가되는 컬럼
    -------------
      rule_pass, rule_violations                 (규칙 기반, 항상 계산)
      faithfulness_score                          (임베딩 유사도, run_semantic=True일 때)
      judge_tone_fit / judge_persuasiveness /
      judge_naturalness / judge_constraint_adherence /
      judge_total / judge_comment                 (LLM judge, run_judge=True일 때)

    debug=True로 실행하면, faithfulness/judge 계산이 실패한 행에 대해
    어떤 모델(model)에서 어떤 예외(exception)가 발생했는지 즉시 출력한다.
    """
    result = df.copy()

    rule_pass_col, rule_violations_col = [], []
    faithfulness_col = []
    judge_cols = {k: [] for k in
                  ["judge_tone_fit", "judge_persuasiveness", "judge_naturalness",
                   "judge_constraint_adherence", "judge_total", "judge_comment"]}

    for _, row in result.iterrows():
        err_val = row.get("error")
        has_error = isinstance(err_val, str) and len(err_val) > 0

        # 1) 규칙 기반은 항상 계산 (API 호출 없음, 비용 0)
        if has_error:
            rule_pass_col.append(None)
            rule_violations_col.append(None)
        else:
            passed, violations = rule_based_check(row, store_info)
            rule_pass_col.append(passed)
            rule_violations_col.append("; ".join(violations) if violations else "")

        # 2) 시맨틱 유사도 (faithfulness_score)
        if has_error or not run_semantic:
            faithfulness_col.append(None)
        else:
            try:
                faithfulness_col.append(faithfulness_score(row, product_attrs, store_info))
            except Exception as e:  # noqa: BLE001
                if debug:
                    print(f"[faithfulness 실패] model={row.get('model')} | "
                          f"{type(e).__name__}: {e}")
                faithfulness_col.append(None)

        # 3) LLM judge
        if has_error or not run_judge:
            for k in judge_cols:
                judge_cols[k].append(None)
        else:
            try:
                score = llm_judge_score(row, product_attrs, store_info, tone, judge_model)
                total = round(
                    (score.tone_fit + score.persuasiveness +
                     score.naturalness + score.constraint_adherence) / 4, 2
                )
                judge_cols["judge_tone_fit"].append(score.tone_fit)
                judge_cols["judge_persuasiveness"].append(score.persuasiveness)
                judge_cols["judge_naturalness"].append(score.naturalness)
                judge_cols["judge_constraint_adherence"].append(score.constraint_adherence)
                judge_cols["judge_total"].append(total)
                judge_cols["judge_comment"].append(score.comment)
            except (ValidationError, json.JSONDecodeError, RuntimeError, Exception) as e:  # noqa: BLE001
                if debug:
                    print(f"[judge 실패] model={row.get('model')} | "
                          f"{type(e).__name__}: {e}")
                for k in judge_cols:
                    judge_cols[k].append(None)

    result["rule_pass"] = rule_pass_col
    result["rule_violations"] = rule_violations_col
    if run_semantic:
        result["faithfulness_score"] = faithfulness_col
    if run_judge:
        for k, v in judge_cols.items():
            result[k] = v

    return result


# ----------------------------------------------------------------------------
# 5. 표시(display) 헬퍼 — judge_comment 등 긴 텍스트가 잘리지 않게 출력
# ----------------------------------------------------------------------------
def show_full(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    judge_comment처럼 긴 텍스트 컬럼이 "..."으로 잘리지 않도록 표시 옵션을 조정한 뒤
    df를 반환한다. (로컬 VSCode/JupyterLab 환경 기준. Colab 전용 코드는 제거함)

    사용법:
        show_full(result)                                      # 전체 컬럼
        show_full(result, columns=["model", "judge_comment"])  # 특정 컬럼만
    """
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    return df[columns] if columns else df


def print_comments(df: pd.DataFrame, comment_col: str = "judge_comment",
                    label_col: str = "model") -> None:
    """
    표 형태 대신, 모델별 judge_comment를 줄글로 하나씩 출력한다.
    표 렌더러 자체의 폭 제한을 아예 우회하고 싶을 때 사용.
    """
    for _, row in df.iterrows():
        comment = row.get(comment_col)
        label = row.get(label_col, "")
        if pd.isna(comment) or not comment:
            continue
        print(f"[{label}] {comment}\n")


# ----------------------------------------------------------------------------
# 6. 단독 실행용 스모크 테스트 (실제 파이프라인은 step1/step6/step7 스크립트를 사용)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from compare_copy_llms import compare_models  # 같은 폴더의 compare_copy_llms.py 필요

    # 파일 의존 없이 바로 실행해볼 수 있도록 인라인 샘플 데이터 사용
    product_attrs = {
        "id": "bakery_croissant_cookie_coldbrew",
        "new_caption": (
            "A wooden plate holding a plain butter croissant, a chocolate-drizzled pastry, "
            "and a chocolate chip cookie, paired with a tall glass of iced cold brew coffee"
            "-warm sunlit cafe setup for a bakery breakfast set."
        ),
    }
    store_info = {"name": "cafe masion", "phone": "02-123-4567", "price": "12,000원"}
    tone = "고급스러운"

    base_df = compare_models(product_attrs, store_info, tone=tone)
    result = evaluate_dataframe(base_df, product_attrs, store_info, tone=tone, debug=True)

    print(show_full(result))
    print_comments(result)

    result.to_csv("eval_copy_smoke_test.csv", index=False, encoding="utf-8-sig")
    print("\n저장 완료 -> eval_copy_smoke_test.csv")
