import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------------
# 기본 설정 및 데이터 주소
# ------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide",
)


# ------------------------------------------------------------
# 데이터 불러오기
# 첫 화면과 같은 주소, UTF-8 인코딩 사용
# ------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔎 데이터 탐색")
st.caption("나이, 평균 혈당, 질환 여부, 체질량지수, 흡연 상태에 따른 데이터를 살펴봅니다.")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.code(str(error))
    st.stop()


required_columns = [
    "age",
    "avg_glucose_level",
    "stroke",
    "hypertension",
    "heart_disease",
    "bmi",
    "smoking_status",
]
missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {', '.join(missing_columns)}")
    st.stop()


# ------------------------------------------------------------
# 1. 나이와 평균 혈당 분포
# ------------------------------------------------------------
st.subheader("1. 나이와 평균 혈당의 분포")
st.write("각 항목의 값이 어떤 구간에 많이 분포하는지 히스토그램으로 확인합니다.")

hist_col1, hist_col2 = st.columns(2)

with hist_col1:
    age_fig = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )
    age_fig.update_layout(yaxis_title="사람 수")
    st.plotly_chart(age_fig, use_container_width=True)

with hist_col2:
    glucose_fig = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"},
    )
    glucose_fig.update_layout(yaxis_title="사람 수")
    st.plotly_chart(glucose_fig, use_container_width=True)


# ------------------------------------------------------------
# 2. 뇌졸중 경험 여부에 따른 나이와 평균 혈당 비교
# ------------------------------------------------------------
st.divider()
st.subheader("2. 뇌졸중 경험 여부에 따른 비교")
st.write("뇌졸중 경험 여부(0: 없음, 1: 있음)에 따라 나이와 평균 혈당을 상자그림으로 비교합니다.")

plot_df = df.loc[df["stroke"].isin([0, 1])].copy()
plot_df["뇌졸중 경험"] = plot_df["stroke"].map({0: "없음", 1: "있음"})

box_col1, box_col2 = st.columns(2)

with box_col1:
    age_box = px.box(
        plot_df,
        x="뇌졸중 경험",
        y="age",
        category_orders={"뇌졸중 경험": ["없음", "있음"]},
        title="뇌졸중 경험 여부에 따른 나이",
        labels={"뇌졸중 경험": "뇌졸중 경험", "age": "나이"},
    )
    st.plotly_chart(age_box, use_container_width=True)

with box_col2:
    glucose_box = px.box(
        plot_df,
        x="뇌졸중 경험",
        y="avg_glucose_level",
        category_orders={"뇌졸중 경험": ["없음", "있음"]},
        title="뇌졸중 경험 여부에 따른 평균 혈당",
        labels={"뇌졸중 경험": "뇌졸중 경험", "avg_glucose_level": "평균 혈당"},
    )
    st.plotly_chart(glucose_box, use_container_width=True)

mean_table = (
    plot_df.groupby("뇌졸중 경험")[["age", "avg_glucose_level"]]
    .mean()
    .reindex(["없음", "있음"])
    .rename(
        columns={
            "age": "평균 나이",
            "avg_glucose_level": "평균 혈당",
        }
    )
    .round(2)
)
st.markdown("**뇌졸중 경험 여부에 따른 평균값**")
st.dataframe(mean_table, use_container_width=True)


# ------------------------------------------------------------
# 3. 고혈압 및 심장병 여부에 따른 뇌졸중 비율
# ------------------------------------------------------------
st.divider()
st.subheader("3. 고혈압·심장병 여부에 따른 뇌졸중 비율")
st.write("각 그룹에서 stroke가 1인 사람의 비율을 계산합니다.")

def make_rate_table(data, group_column):
    grouped = data.groupby(group_column)["stroke"].agg(["mean", "count"])
    grouped = grouped.reindex([0, 1])
    result = pd.DataFrame(
        {
            "구분": ["없음", "있음"],
            "뇌졸중 비율(%)": grouped["mean"].to_numpy() * 100,
            "사람 수": grouped["count"].fillna(0).astype(int).to_numpy(),
        }
    )
    return result


hypertension_rates = make_rate_table(df, "hypertension")
heart_disease_rates = make_rate_table(df, "heart_disease")

rate_col1, rate_col2 = st.columns(2)

with rate_col1:
    hypertension_fig = px.bar(
        hypertension_rates,
        x="구분",
        y="뇌졸중 비율(%)",
        text=hypertension_rates["뇌졸중 비율(%)"].map(
            lambda value: f"{value:.2f}%" if pd.notna(value) else "자료 없음"
        ),
        title="고혈압 여부에 따른 뇌졸중 비율",
        labels={"구분": "고혈압", "뇌졸중 비율(%)": "뇌졸중 비율(%)"},
    )
    hypertension_fig.update_traces(textposition="outside")
    hypertension_fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(hypertension_fig, use_container_width=True)

with rate_col2:
    heart_fig = px.bar(
        heart_disease_rates,
        x="구분",
        y="뇌졸중 비율(%)",
        text=heart_disease_rates["뇌졸중 비율(%)"].map(
            lambda value: f"{value:.2f}%" if pd.notna(value) else "자료 없음"
        ),
        title="심장병 여부에 따른 뇌졸중 비율",
        labels={"구분": "심장병", "뇌졸중 비율(%)": "뇌졸중 비율(%)"},
    )
    heart_fig.update_traces(textposition="outside")
    heart_fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(heart_fig, use_container_width=True)


# ------------------------------------------------------------
# 4. BMI 결측 그룹과 전체의 뇌졸중 비율
# ------------------------------------------------------------
st.divider()
st.subheader("4. 체질량지수(bmi)가 비어 있는 사람")
st.write("BMI가 비어 있는 사람만 따로 구분해, 전체 데이터의 뇌졸중 비율과 비교합니다.")

bmi_missing = df["bmi"].isna()
missing_bmi_df = df.loc[bmi_missing]

missing_bmi_count = len(missing_bmi_df)
missing_bmi_stroke_count = int(missing_bmi_df["stroke"].eq(1).sum())
all_stroke_count = int(df["stroke"].eq(1).sum())

missing_bmi_rate = (
    missing_bmi_stroke_count / missing_bmi_count * 100
    if missing_bmi_count
    else float("nan")
)
overall_rate = df["stroke"].mean() * 100 if len(df) else float("nan")

bmi_comparison = pd.DataFrame(
    [
        {
            "구분": "BMI 빈 값인 사람",
            "사람 수": missing_bmi_count,
            "뇌졸중 경험자 수": missing_bmi_stroke_count,
            "뇌졸중 비율(%)": missing_bmi_rate,
        },
        {
            "구분": "전체 사람",
            "사람 수": len(df),
            "뇌졸중 경험자 수": all_stroke_count,
            "뇌졸중 비율(%)": overall_rate,
        },
    ]
)
bmi_comparison["뇌졸중 비율(%)"] = bmi_comparison["뇌졸중 비율(%)"].round(2)
st.dataframe(bmi_comparison, use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# 5. 흡연 상태별 사람 수
# ------------------------------------------------------------
st.divider()
st.subheader("5. 흡연 상태별 사람 수")
st.write("흡연 상태의 각 값에 해당하는 사람 수를 셉니다. 비어 있는 값이 있다면 '결측'으로 표시합니다.")

smoking_counts = (
    df["smoking_status"]
    .fillna("결측")
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking_counts, use_container_width=True, hide_index=True)

st.caption("※ 이 페이지는 데이터 탐색 및 학습용입니다. 통계적 차이가 개인의 뇌졸중 위험이나 원인을 확정하지는 않습니다.")
