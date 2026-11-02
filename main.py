from pathlib import Path
import json

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
RECORD_DIR = Path(__file__).resolve().parent / "내_기록"
RECORD_FILE = RECORD_DIR / "데이터_설명.json"

st.set_page_config(
    page_title=APP_TITLE,   # 브라우저 탭 제목
    page_icon="🧠",
    layout="wide",
)


# ------------------------------------------------------------
# 교재 내용을 저장하고 불러오는 함수
# ------------------------------------------------------------
def load_notes():
    """저장된 우리말 뜻과 데이터 출처를 불러옵니다."""
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        return {
            "meanings": saved.get("meanings", {}),
            "source": saved.get("source", ""),
        }
    except (json.JSONDecodeError, OSError):
        st.warning("기존 기록 파일을 읽지 못해 빈 기록으로 시작합니다.")
        return {"meanings": {}, "source": ""}


def save_notes(meanings, source):
    """우리말 뜻과 출처를 UTF-8 JSON 파일로 저장합니다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meanings": meanings,
        "source": source,
    }
    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고 각 열의 의미를 교재와 함께 정리해 보는 공간입니다.")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.code(str(error))
    st.stop()


# ------------------------------------------------------------
# 데이터 요약 카드
# ------------------------------------------------------------
people_count = len(df)
column_count = len(df.columns)
stroke_count = int(df["stroke"].eq(1).sum()) if "stroke" in df.columns else 0
stroke_rate = (stroke_count / people_count * 100) if people_count else 0

st.subheader("📊 데이터 한눈에 보기")
metric_cols = st.columns(4)
metric_cols[0].metric("전체 사람 수", f"{people_count:,}명")
metric_cols[1].metric("열 개수", f"{column_count:,}개")
metric_cols[2].metric("뇌졸중 경험자 수", f"{stroke_count:,}명")
metric_cols[3].metric("뇌졸중 경험자 비율", f"{stroke_rate:.2f}%")


# ------------------------------------------------------------
# 열 정보 표와 교재 내용 입력
# ------------------------------------------------------------
st.divider()
st.subheader("🗂️ 데이터 열 살펴보기")
st.write(
    "각 열의 이름과 값의 형태, 빈 값 개수를 확인하세요. "
    "**우리말 뜻**은 교재를 참고해 직접 입력한 뒤 아래 저장 버튼을 눌러 주세요."
)

saved_notes = load_notes()
saved_meanings = saved_notes["meanings"]

column_info = []
for column in df.columns:
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        value_kind = f"숫자형 (고유값 {series.nunique(dropna=True):,}종)"
    else:
        value_kind = f"문자형 (고유값 {series.nunique(dropna=True):,}종)"

    column_info.append(
        {
            "열 이름": column,
            "우리말 뜻": saved_meanings.get(column, ""),
            "값의 종류": value_kind,
            "빈 값 개수": int(series.isna().sum()),
        }
    )

info_df = pd.DataFrame(
    column_info,
    columns=["열 이름", "우리말 뜻", "값의 종류", "빈 값 개수"],
)

with st.form("notes_form"):
    edited_info = st.data_editor(
        info_df,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "열 이름": st.column_config.TextColumn("열 이름"),
            "우리말 뜻": st.column_config.TextColumn(
                "우리말 뜻",
                help="교재를 참고해 각 열의 뜻을 입력하세요.",
                width="large",
            ),
            "값의 종류": st.column_config.TextColumn("값의 종류"),
            "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
        },
    )

    st.markdown("#### 📚 데이터 출처")
    source_text = st.text_area(
        "교재에 있는 데이터 출처 내용을 입력하세요.",
        value=saved_notes["source"],
        height=100,
        placeholder="데이터 출처를 여기에 적어 주세요.",
    )

    submitted = st.form_submit_button("💾 우리말 뜻과 출처 저장", type="primary")

    if submitted:
        meanings_to_save = {
            str(row["열 이름"]): str(row["우리말 뜻"] or "")
            for _, row in edited_info.iterrows()
        }
        try:
            save_notes(meanings_to_save, source_text)
            st.success(f"기록을 저장했습니다. 저장 위치: {RECORD_FILE}")
        except OSError as error:
            st.error(f"기록을 저장하지 못했습니다: {error}")


# ------------------------------------------------------------
# 원본 데이터 미리보기
# ------------------------------------------------------------
st.divider()
st.subheader("🔎 원본 데이터 미리보기")
st.write("데이터의 처음 다섯 줄을 원래 열과 값 그대로 표시합니다.")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

st.caption("※ 이 화면은 데이터 소개용입니다. 뇌졸중 진단이나 의학적 판단을 위한 도구가 아닙니다.")
