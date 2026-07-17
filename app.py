import streamlit as st
from pathlib import Path
# 프로젝트 내부 모듈 임포트 (실제 디렉토리 구조에 맞게 수정 필요)
from adcg.pipeline import run_pipeline

st.set_page_config(page_title="AI 광고 생성기", layout="wide")

# 1. 페이지 설정 및 디자인(CSS) 적용
st.set_page_config(page_title="AI 로 만드는 우리 가게 광고", layout="wide")

st.markdown("""
    <style>
    /* 전체 배경색 */
    .stApp {
        background-color: #FFF8E7; 
    }
    
    /* 제목 */
    h1 {
        color: #5D4037; 
    }
    
    /* 버튼 */
    div.stButton > button {
        background-color: #FFB347;
        color: white;
        border: none;
    }

    /* 사이드바 배경색 */
    [data-testid="stSidebar"] {
        background-color: #FFE4C4; 
    }
    
    /* 사이드바 텍스트 색상 */
    [data-testid="stSidebar"] * {
        color: #5D4037;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. 제목
st.title("AI 로 만드는 우리 가게 광고")

# 3. 경로 설정 및 강제 생성
OUTPUT_DIR = Path("outputs/pipeline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # 여기서 폴더를 확실하게 생성합니다.

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    gpt_model = st.selectbox("GPT 모델 선택", ["gpt-5.4-nano", "gpt-4"])
    copy_count = st.number_input("광고 카피 수", min_value=1, max_value=20, value=9)
    layout_mode = st.radio("레이아웃 모드", ["layout", "preserve"])
    seed = st.number_input("Seed 값", value=42)

# 메인 UI
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("상품 이미지 업로드", type=["jpg", "png"])
    product_info = st.text_area("매장 및 상품 정보 입력")

with col2:
    if uploaded_file and product_info:
        if st.button("광고 생성 시작"):
            with st.spinner("광고 생성 중..."):
                try:
                    # 2. 파이프라인 실행 (여기에 실제 입력값이 들어가는지 확인하세요)
                    # 예: result = run_pipeline(image_path=..., info_path=..., output_dir=OUTPUT_DIR)
                    
                    # 주의: 아직 run_pipeline을 연결하지 않았다면 여기에 코드를 넣어야 합니다.
                    # result = run_pipeline(...)
                    
                    st.success("생성 완료!")
                    st.write(f"파일이 저장된 위치: {OUTPUT_DIR.absolute()}")
                    
                except Exception as e:
                    # 3. 에러 발생 시 화면에 표시
                    st.error(f"오류가 발생했습니다: {e}")
    else:
        st.info("이미지와 정보를 모두 입력해주세요.")