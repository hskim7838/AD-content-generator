import streamlit as st
import os
import json
from pathlib import Path

# 팀 프로젝트 내부 파이프라인 모듈 임포트
from adcg.pipeline import run_pipeline 

# 1. 페이지 설정
st.set_page_config(page_title="AI 로 만드는 우리 가게 광고", layout="wide")

# 2. PPT 29-34 톤앤매너 프리미엄 CSS 적용
st.markdown(
"""
<style>
/* 전체 앱 및 사이드바 배경 (웜 화이트 & 소프트 베이지) */
.stApp {
    background-color: #FDFBF7;
}
[data-testid="stSidebar"] {
    background-color: #F4EBE1 !important;
    border-right: 1px solid #E6DDD0;
}

/* 타이틀 및 텍스트 가독성 고정 (딥 차콜) */
h1, h2, h3, h4, h5, h6, p, span, label {
    color: #2B2B2B !important;
    font-family: 'Noto Sans KR', sans-serif;
}

/* STEP 포인트 라벨 (테라코타 오렌지) */
h4[id^="step"] {
    color: #D35400 !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
}

/* 입력 필드 디자인 (화이트 박스 + 세련된 테두리) */
[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D1C7BD !important;
    border-radius: 8px !important;
}

/* 입력 필드 포커스 시 오렌지 테두리 */
[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover,
[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
    border-color: #D35400 !important;
}

/* 입력창 내부 글자색 완벽 고정 (가독성 확보) */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #2B2B2B !important;
    -webkit-text-fill-color: #2B2B2B !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] svg {
    fill: #6E655B !important;
}

/* 상단 헤더 투명화 */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* 실행 버튼 디자인 (포인트 테라코타 오렌지) */
div.stButton > button {
    background-color: #D35400 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: bold !important;
    box-shadow: 0 2px 4px rgba(211, 84, 0, 0.2) !important;
}
div.stButton > button:hover {
    background-color: #E67E22 !important;
}
div.stButton > button p {
    color: #FFFFFF !important;
}

/* 파일 업로더 내 버튼 */
[data-testid="stFileUploader"] button {
    background-color: #2B2B2B !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
}

/* 안내 메시지 상자 */
div.stWarning, div.stInfo {
    background-color: #F4EBE1 !important;
    border: 1px solid #D1C7BD !important;
    border-radius: 8px !important;
}
div.stWarning *, div.stInfo * {
    color: #D35400 !important;
}
</style>
""", 
unsafe_allow_html=True)

# 3. 입출력 경로 및 폴더 설정
INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs/test_run")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 4. 세션 상태 초기화 (현재 단계 기억)
if 'step' not in st.session_state:
    st.session_state.step = 1

# 5. 글로벌 고정 사이드바 설정 구역
with st.sidebar:
    st.header("⚙️ 광고 생성 설정")
    gpt_model = st.selectbox("GPT 모델 선택", ["gpt-5.4-nano", "gpt-5.4-mini"])
    layout_mode = st.radio("레이아웃 모드", ["layout", "preserve"])
# 슬라이더 코드: 사이드바 입력부에 추가
    focus_strength = st.slider(
        "광고 강조 강도 (Focus Strength)", 
        min_value=0.0, 
        max_value=1.0, 
        value=1.0, 
        step=0.1
        )

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
            
            # 사용자로부터 직접 입력받는 폼 생성 (예시 텍스트를 기본값으로 삽입)
            product_name = st.text_input("상품명", value="수제 베이커리 세트")
            product_description = st.text_area("상품 설명", value="다양한 빵과 쿠키, 아이스커피로 구성된 세트")
            seller_description = st.text_area("판매자 설명", value="매일 아침 직접 굽는 동네 베이커리")
            store_info = st.text_input("매장 정보", value="따뜻하고 편안한 분위기의 소규모 카페")
            
            # 리뷰는 여러 줄로 입력받아 리스트로 변환하기 쉽도록 안내
            reviews_input = st.text_area("리뷰 (엔터로 구분해서 입력)", value="빵이 부드럽고 신선해요\n커피와 함께 먹기 좋아요")
            
            focus = st.selectbox("광고 포커스 선택", options=["product", "brand", "sales"])
            additional_request = st.text_area("추가 요청사항", value="따뜻한 아침 햇살이 들어오는 카페 분위기")

    st.divider()
    
    if st.button("광고 제작 시작하기 ➔", use_container_width=True):
        if uploaded_image:
            # 1. 이미지 파일을 로컬 디렉토리에 보존
            img_path = INPUT_DIR / uploaded_image.name
            
            with open(img_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            # 2. 입력받은 데이터들을 딕셔너리로 묶고 리스트 형태 정리
            reviews_list = [r.strip() for r in reviews_input.split('\n') if r.strip()]
                
            info_data = {
                "product_name": product_name,
                "product_description": product_description,
                "seller_description": seller_description,
                "store_info": store_info,
                "reviews": reviews_list,
                "focus": focus,
                "additional_request": additional_request
            }
                
            # 3. 딕셔너리를 JSON 파일로 자동 생성하여 저장
            json_path = INPUT_DIR / "user_input_info.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(info_data, f, ensure_ascii=False, indent=4)
                    
            # 다음 단계를 위해 세션 상태에 저장
            st.session_state.img_path = img_path
            st.session_state.json_path = json_path
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
            # 사이드바에서 설정한 핵심 파라미터들을 실제 run_pipeline에 동적 전달
            result = run_pipeline(
                image_path=st.session_state.img_path, 
                info_path=st.session_state.json_path, 
                output_dir=OUTPUT_DIR,
                gpt_model=gpt_model,
                layout_mode=layout_mode,
                seed=42,
                focus_strength=focus_strength  # focus_strength 슬라이더 UI 추가
            )
            
            # 파이프라인 결과 객체(PipelineResult)에 들어있는 최종 렌더링 이미지 경로 획득
            if hasattr(result, 'final_image') and result.final_image:
                st.session_state.final_image_path = Path(result.final_image)
            
            st.session_state.step = 3
            st.rerun()
            
        except Exception as e:
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
        output_files = list(OUTPUT_DIR.rglob("*.png")) + list(OUTPUT_DIR.rglob("*.jpg")) + list(OUTPUT_DIR.rglob("*.jpeg"))
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
        if 'final_image_path' in st.session_state:
            del st.session_state.final_image_path
        st.rerun()