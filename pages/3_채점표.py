import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ------------------------------------------------------------
# 기본 설정: 분류 모델 페이지와 같은 데이터 및 모델 설정
# ------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
RANDOM_STATE = 42

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

st.set_page_config(
    page_title="채점표 | 뇌졸중 예측 실습실",
    page_icon="🧾",
    layout="wide",
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🧾 채점표")
st.caption("테스트 데이터로 모델의 정답과 오답을 확인하고, 놓친 환자와 헛짚은 사람을 살펴봅니다.")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.code(str(error))
    st.stop()

required_columns = ["id", "stroke"] + list(FEATURE_LABELS)
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {', '.join(missing_columns)}")
    st.stop()


# ------------------------------------------------------------
# 분류 모델 페이지와 같은 분할: ID 정렬, 10명 중 앞 3명 테스트
# ------------------------------------------------------------
df = df.sort_values("id", kind="mergesort").reset_index(drop=True)
test_mask = [position % 10 < 3 for position in range(len(df))]
train_df = df.loc[[not value for value in test_mask]].copy()
test_df = df.loc[test_mask].copy()

st.subheader("테스트 데이터")
test_stroke_count = int(test_df["stroke"].eq(1).sum())
st.markdown(
    f"테스트용 사람 수: **{len(test_df):,}명**　|　"
    f"그중 실제 뇌졸중인 사람: **{test_stroke_count:,}명**"
)
if len(test_df) != 1533:
    st.warning(
        f"현재 테스트용 인원은 {len(test_df):,}명입니다. "
        "요청한 1,533명과 다르면 원본 데이터의 행 수를 확인해 주세요."
    )


# ------------------------------------------------------------
# 입력 속성: 분류 모델 페이지의 기본 선택과 동일
# ------------------------------------------------------------
st.subheader("모델 입력 속성")
selected_features = st.multiselect(
    "두 모델에 사용할 속성을 선택하세요.",
    options=list(FEATURE_LABELS),
    default=["age", "avg_glucose_level", "hypertension", "heart_disease"],
    format_func=lambda feature: FEATURE_LABELS[feature],
)

if len(selected_features) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요.")
    st.stop()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()
y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

# BMI를 선택한 경우에만 학습용 중앙값으로 결측값 대체
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()
    if pd.isna(bmi_median):
        st.error("학습용 데이터에서 체질량지수(bmi)의 중앙값을 계산할 수 없습니다.")
        st.stop()
    train_missing = int(X_train["bmi"].isna().sum())
    test_missing = int(X_test["bmi"].isna().sum())
    X_train.loc[:, "bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test.loc[:, "bmi"] = X_test["bmi"].fillna(bmi_median)
    st.caption(
        f"체질량지수의 빈 값은 학습용 중앙값 {bmi_median:.2f}으로 채웠습니다 "
        f"(학습용 {train_missing:,}명, 테스트용 {test_missing:,}명)."
    )
else:
    st.caption("체질량지수(bmi)를 선택하지 않아 BMI 빈 값은 처리하지 않았습니다.")

remaining_missing = X_train.isna().sum() + X_test.isna().sum()
remaining_missing = remaining_missing[remaining_missing > 0]
if not remaining_missing.empty:
    missing_text = ", ".join(
        f"{FEATURE_LABELS[column]} {int(count)}개"
        for column, count in remaining_missing.items()
    )
    st.error(f"선택한 속성에 처리되지 않은 빈 값이 있습니다: {missing_text}")
    st.stop()

for column in selected_features:
    X_train.loc[:, column] = pd.to_numeric(X_train[column], errors="raise")
    X_test.loc[:, column] = pd.to_numeric(X_test[column], errors="raise")


# ------------------------------------------------------------
# 두 모델 학습: 분류 모델 페이지와 동일한 설정
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
)
logistic_model.fit(X_train_scaled, y_train)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
tree_model.fit(X_train, y_train)

# 학습용 다수결 기준 모델
majority_class = int(y_train.value_counts().idxmax())

model_predictions = {
    "로지스틱 회귀(확률로 답하는 모델)": {
        "train": logistic_model.predict(X_train_scaled),
        "test": logistic_model.predict(X_test_scaled),
    },
    "의사결정트리(질문으로 답하는 모델)": {
        "train": tree_model.predict(X_train),
        "test": tree_model.predict(X_test),
    },
    "훈련용에서 많은 쪽으로만 답하는 모델": {
        "train": [majority_class] * len(y_train),
        "test": [majority_class] * len(y_test),
    },
}


# ------------------------------------------------------------
# 혼동행렬 및 지표 계산
# 양성 = stroke 1
# ------------------------------------------------------------
def confusion_counts(actual, predicted):
    actual = pd.Series(actual).astype(int).reset_index(drop=True)
    predicted = pd.Series(predicted).astype(int).reset_index(drop=True)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def calculate_metrics(counts, total):
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]

    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) else None

    if tp + fp == 0:
        precision = None
        f1 = None
    else:
        precision = tp / (tp + fp)
        if recall is None or precision + recall == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1,
    }


def format_metric(value):
    return "계산할 수 없음" if value is None else f"{value * 100:.2f}%"


def make_confusion_table(counts):
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    return pd.DataFrame(
        [
            {
                "실제 \\ 예측": "아님(0)",
                "아님(0)": f"TN: {tn:,}명",
                "뇌졸중(1)": f"FP: {fp:,}명",
                "행 합계": f"{tn + fp:,}명",
            },
            {
                "실제 \\ 예측": "뇌졸중(1)",
                "아님(0)": f"FN: {fn:,}명",
                "뇌졸중(1)": f"TP: {tp:,}명",
                "행 합계": f"{fn + tp:,}명",
            },
            {
                "실제 \\ 예측": "열 합계",
                "아님(0)": f"{tn + fn:,}명",
                "뇌졸중(1)": f"{fp + tp:,}명",
                "행 합계": f"{tn + fp + fn + tp:,}명",
            },
        ]
    )


test_counts = {}
metric_rows = []

for model_name, predictions in model_predictions.items():
    counts = confusion_counts(y_test, predictions["test"])
    test_counts[model_name] = counts
    metrics = calculate_metrics(counts, len(y_test))
    metric_rows.append(
        {
            "모델": model_name,
            "정확도": format_metric(metrics["정확도"]),
            "재현율": format_metric(metrics["재현율"]),
            "정밀도": format_metric(metrics["정밀도"]),
            "F1": format_metric(metrics["F1"]),
        }
    )


# ------------------------------------------------------------
# 모델별 혼동행렬
# ------------------------------------------------------------
st.divider()
st.subheader("1. 테스트 데이터 혼동행렬")
st.write(
    "행은 실제 결과, 열은 모델의 예측입니다. "
    "양성은 뇌졸중(1)이며 각 칸에 TP·FP·FN·TN 약자와 사람 수를 함께 표시합니다."
)

matrix_columns = st.columns(3)
for container, model_name in zip(matrix_columns, model_predictions):
    with container:
        st.markdown(f"**{model_name}**")
        st.dataframe(
            make_confusion_table(test_counts[model_name]),
            use_container_width=True,
            hide_index=True,
        )


# ------------------------------------------------------------
# 네 가지 평가 지표 비교
# ------------------------------------------------------------
st.divider()
st.subheader("2. 네 가지 평가 지표 비교")
st.write(
    "정확도는 전체 테스트 인원 중 맞힌 비율, 재현율은 실제 뇌졸중인 사람 중 "
    "찾아낸 비율, 정밀도는 뇌졸중이라고 예측한 사람 중 실제 뇌졸중인 비율입니다."
)
metrics_df = pd.DataFrame(metric_rows)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)
st.caption(
    "F1은 정밀도와 재현율의 조화평균입니다. 뇌졸중이라고 예측한 사람이 0명이면 "
    "정밀도와 F1은 '계산할 수 없음'으로 표시합니다."
)


# ------------------------------------------------------------
# 놓친 환자(FN)와 헛짚은 사람(FP) 목록
# ------------------------------------------------------------
st.divider()
st.subheader("3. 놓친 환자와 헛짚은 사람")
selected_model = st.selectbox(
    "목록을 확인할 모델을 선택하세요.",
    options=list(model_predictions),
)

selected_predictions = pd.Series(
    model_predictions[selected_model]["test"],
    index=test_df.index,
    name="예측",
).astype(int)

actual_test = test_df["stroke"].astype(int)

# 실제 1인데 예측 0: FN
missed_mask = (actual_test == 1) & (selected_predictions == 0)
# 실제 0인데 예측 1: FP
false_alarm_mask = (actual_test == 0) & (selected_predictions == 1)

display_columns = ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]
korean_columns = {
    "id": "번호",
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부",
}


def make_people_list(mask):
    people = test_df.loc[mask, display_columns].copy()
    people = people.sort_values("age", ascending=False, kind="mergesort")
    people = people.rename(columns=korean_columns)
    people["고혈압 여부"] = people["고혈압 여부"].map(
        {0: "없음", 1: "있음"}
    ).fillna(people["고혈압 여부"].astype(str))
    people["심장병 여부"] = people["심장병 여부"].map(
        {0: "없음", 1: "있음"}
    ).fillna(people["심장병 여부"].astype(str))
    people["나이"] = people["나이"].round(1)
    people["평균 혈당"] = people["평균 혈당"].round(2)
    return people


missed_people = make_people_list(missed_mask)
false_alarm_people = make_people_list(false_alarm_mask)

list_col1, list_col2 = st.columns(2)

with list_col1:
    st.markdown(f"**놓친 환자 (FN): {len(missed_people):,}명**")
    st.caption("실제 뇌졸중인데 모델이 '아님'이라고 예측한 사람")
    st.dataframe(
        missed_people.head(20),
        use_container_width=True,
        hide_index=True,
    )
    if len(missed_people) > 20:
        st.caption(f"전체 {len(missed_people):,}명 중 나이가 많은 순서로 앞 20명만 표시했습니다.")

with list_col2:
    st.markdown(f"**헛짚은 사람 (FP): {len(false_alarm_people):,}명**")
    st.caption("실제 뇌졸중이 아닌데 모델이 '뇌졸중'이라고 예측한 사람")
    st.dataframe(
        false_alarm_people.head(20),
        use_container_width=True,
        hide_index=True,
    )
    if len(false_alarm_people) > 20:
        st.caption(f"전체 {len(false_alarm_people):,}명 중 나이가 많은 순서로 앞 20명만 표시했습니다.")

selected_counts = test_counts[selected_model]
st.markdown(
    f"**목록 인원 확인:** 놓친 환자 {len(missed_people):,}명 = 혼동행렬 FN "
    f"{selected_counts['FN']:,}명, 헛짚은 사람 {len(false_alarm_people):,}명 = "
    f"혼동행렬 FP {selected_counts['FP']:,}명입니다."
)

st.caption(
    "※ 이 채점표는 학습용 데이터 분석 결과입니다. 실제 의료 진단이나 치료 결정을 대신하지 않습니다."
)
