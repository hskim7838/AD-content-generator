import streamlit as st
import json
import os
import time
from pathlib import Path
from uuid import uuid4

# 팀 프로젝트 내부 파이프라인 모듈 임포트
from adcg.generation import GENERATION_DEFAULTS, load_generation_pipeline
from adcg.runtime import InferenceQueue

# 1. 페이지 설정
st.set_page_config(page_title="AI 로 만드는 우리 가게 광고", layout="wide")


APP_DIR = Path(__file__).resolve().parent
CSS_PATH = APP_DIR / "assets" / "styles.css"


def load_css():
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_inference_queue():
    def load_pipe():
        return load_generation_pipeline(
            base_model=GENERATION_DEFAULTS["base_model"],
            controlnet_model=GENERATION_DEFAULTS["controlnet_model"],
            cpu_offload=False,
        )

    return InferenceQueue(pipe_factory=load_pipe)


# 첫 화면을 막지 않고 GPU Worker에서 모델 로드를 시작한다.
inference_queue = get_inference_queue()

# 2. PPT 29-34 톤앤매너 프리미엄 CSS 적용
load_css()

# 3. 입출력 경로 및 폴더 설정
INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs/test_run")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 4. 세션 상태 초기화 (현재 단계 기억)
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'session_id' not in st.session_state:
    st.session_state.session_id = uuid4().hex

SESSION_INPUT_DIR = INPUT_DIR / st.session_state.session_id
SESSION_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# 5. 글로벌 고정 사이드바 설정 구역
with st.sidebar:
    st.header("⚙️ 광고 생성 설정")
    if inference_queue.model_status == "loading":
        st.caption("🟡 GPU 모델을 백그라운드에서 준비하고 있습니다.")
    elif inference_queue.model_status == "ready":
        st.caption("🟢 GPU 모델 준비 완료")
    elif inference_queue.model_status == "failed":
        st.error(
            "GPU 모델 준비 실패: "
            f"{inference_queue.model_error}"
        )
    gpt_model = st.selectbox("GPT 모델 선택", ["gpt-5.4-nano", "gpt-4"])
    copy_count = st.number_input("광고 카피 수", min_value=1, max_value=20, value=9)
    focus_strength = st.slider(
        "상품 강조 강도",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="높을수록 상품이 더 선명하고 배경이 단순하게 표현됩니다.",
    )
    layout_mode = st.radio("레이아웃 모드", ["layout", "preserve"])
    seed = st.number_input("Seed 값", value=42)

# --- STEP 1: 정보 및 파일 업로드 단계 ---
if st.session_state.step == 1:
    st.markdown("<h4 id='step1'>STEP 01 - BASIC INPUT</h4>", unsafe_allow_html=True)
    st.title("사장님의 상품 사진과 매장 정보를 올려주세요")
    st.caption("업로드된 데이터는 로컬 환경에 안전하게 임시 저장되어 AI 파이프라인의 인풋으로 사용됩니다.")
    st.divider()

    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("A. 상품 이미지 업로드")
        uploaded_image = st.file_uploader("이미지 파일 선택 (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            st.image(uploaded_image, caption="업로드 이미지 미리보기", use_column_width=True)
            
    with col_right:
        st.subheader("B. 매장 정보 직접 입력")
        product_name = st.text_input(
            "상품명",
            value="수제 베이커리 세트",
        )
        product_description = st.text_area(
            "상품 설명",
            value="다양한 빵과 쿠키, 아이스커피로 구성된 세트",
        )
        seller_description = st.text_area(
            "판매자 설명",
            value="매일 아침 직접 굽는 동네 베이커리",
        )
        store_info = st.text_input(
            "매장 정보",
            value="따뜻하고 편안한 분위기의 소규모 카페",
        )
        reviews_input = st.text_area(
            "리뷰 (엔터로 구분해서 입력)",
            value="빵이 부드럽고 신선해요\n커피와 함께 먹기 좋아요",
        )
        focus = st.selectbox(
            "광고 포커스 선택",
            options=["product", "brand", "sales"],
        )
        additional_request = st.text_area(
            "추가 요청사항",
            value="따뜻한 아침 햇살이 들어오는 카페 분위기",
        )

    st.divider()
    
    if st.button("광고 제작 시작하기 ➔", use_container_width=True):
        if uploaded_image:
            # 이미지와 자동 생성한 입력 JSON을 세션별 폴더에 보존
            img_path = SESSION_INPUT_DIR / Path(uploaded_image.name).name
            json_path = SESSION_INPUT_DIR / "user_input_info.json"
            
            with open(img_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            reviews = [
                review.strip()
                for review in reviews_input.splitlines()
                if review.strip()
            ]
            info_data = {
                "product_name": product_name.strip(),
                "product_description": product_description.strip(),
                "seller_description": seller_description.strip(),
                "store_info": store_info.strip(),
                "reviews": reviews,
                "focus": focus,
                "additional_request": additional_request.strip(),
            }
            json_path.write_text(
                json.dumps(info_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            
            # 다음 단계를 위해 세션 상태에 저장
            st.session_state.img_path = img_path
            st.session_state.json_path = json_path
            st.session_state.direction = (
                "brand_focus" if focus == "brand" else "product_focus"
            )

            job_id = uuid4().hex
            job_output_dir = OUTPUT_DIR / job_id
            submission = inference_queue.submit(
                job_id=job_id,
                pipeline_kwargs={
                    "image_path": img_path,
                    "info_path": json_path,
                    "output_dir": job_output_dir,
                    "gpt_model": gpt_model,
                    "copy_count": int(copy_count),
                    "direction": st.session_state.direction,
                    "focus_strength": float(focus_strength),
                    "layout_mode": layout_mode,
                    "seed": int(seed),
                },
            )
            st.session_state.job_id = job_id
            st.session_state.job_output_dir = job_output_dir
            st.session_state.pipeline_submission = submission
            
            st.session_state.step = 2
            st.rerun()
        else:
            st.warning("⚠️ 상품 이미지 파일을 업로드해야 실행할 수 있습니다.")

# --- STEP 2: 파이프라인 구동 단계 (VRAM 연산) ---
elif st.session_state.step == 2:
    st.markdown("<h4 id='step2'>STEP 02 - GENERATION</h4>", unsafe_allow_html=True)
    st.title("AI 파이프라인 연산 가동 중 ⏳")
    
    with st.spinner("고대역폭 VRAM 인프라를 활용하여 이미지 분석, 카피라이팅 및 레이아웃 배치 중..."):
        try:
            submission = st.session_state.get("pipeline_submission")

            if submission is None:
                st.error("등록된 작업 정보를 찾을 수 없습니다.")
                if st.button("⬅ 처음으로 돌아가기"):
                    st.session_state.step = 1
                    st.rerun()
                st.stop()

            st.caption(f"작업 ID: `{submission.job_id}`")

            if not submission.future.done():
                if inference_queue.model_status == "loading":
                    st.info(
                        "GPU 모델을 준비하고 있습니다. 준비가 끝나면 "
                        "등록된 작업을 자동으로 시작합니다."
                    )
                elif submission.future.running():
                    st.info("GPU Worker가 이미지를 생성하고 있습니다.")
                else:
                    st.info(
                        f"작업 대기 중입니다. 등록 당시 앞에 "
                        f"{submission.queued_ahead}개의 작업이 있었습니다."
                    )

                time.sleep(1)
                st.rerun()

            result = submission.future.result()
            del st.session_state.pipeline_submission
            
            # 파이프라인 결과 객체(PipelineResult)에 들어있는 최종 렌더링 이미지 경로 획득
            if hasattr(result, 'final_image') and result.final_image:
                st.session_state.final_image_path = Path(result.final_image)
            
            st.session_state.step = 3
            st.rerun()
            
        except Exception as e:
            st.session_state.pop("pipeline_submission", None)
            st.error(f"파이프라인 실행 중 오류가 발생했습니다: {e}")
            if st.button("⬅ 처음으로 돌아가기"):
                st.session_state.step = 1
                st.rerun()

# --- STEP 3: 최신 결과 표출 단계 ---
elif st.session_state.step == 3:
    st.markdown("<h4 id='step3'>STEP 03 - RESULT</h4>", unsafe_allow_html=True)
    st.title("🎉 완성된 광고 디스플레이")
    st.divider()
    
    # 1순위: 세션에 기록된 정확한 파이프라인 최종 출력 파일 확인
    if 'final_image_path' in st.session_state and st.session_state.final_image_path.exists():
        final_img = st.session_state.final_image_path
    else:
        # 2순위 백업: 깊은 하위 폴더(07_prompt_layout 등)까지 뒤져서 가장 최근에 수정된 이미지 매핑
        result_dir = Path(
            st.session_state.get("job_output_dir", OUTPUT_DIR)
        )
        output_files = list(result_dir.rglob("*.png")) + list(result_dir.rglob("*.jpg")) + list(result_dir.rglob("*.jpeg"))
        final_img = max(output_files, key=os.getmtime) if output_files else None

    if final_img:
        col1, col2 = st.columns([6, 4])
        with col1:
            st.image(str(final_img), use_column_width=True, caption="AI Generated Ad Creative")
        with col2:
            st.success("✅ 고해상도 광고 시안 빌드가 성공적으로 끝났습니다.")
            st.info(f"📁 **로컬 저장소 경로:**\n{final_img.absolute()}")
            
            # 다운로드 인터페이스 공급
            with open(final_img, "rb") as file:
                st.download_button(
                    label="광고 이미지 다운로드 📥",
                    data=file,
                    file_name=final_img.name,
                    mime="image/png",
                    use_container_width=True
                )
    else:
        st.error("최종 출력 이미지를 저장소 시스템에서 식별하지 못했습니다. 디렉토리 구조를 확인하세요.")
    
    st.divider()
    if st.button("🔄 새로운 광고 만들기 (처음으로)", use_container_width=True):
        st.session_state.step = 1
        st.session_state.pop("final_image_path", None)
        st.session_state.pop("job_output_dir", None)
        st.session_state.pop("job_id", None)
        st.session_state.pop("direction", None)
        st.session_state.pop("pipeline_submission", None)
        st.rerun()
