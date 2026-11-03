import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ------------------------------------------------------------
# 기본 설정
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

LOGISTIC_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"

st.set_page_config(
    page_title="성능 높이기 | 뇌졸중 예측 실습실",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("⚖️ 성능 높이기")
st.caption(
    "뇌졸중을 겪은 사람이 적다는 점을 고려해, "
    "클래스별 가중치를 맞추기 전과 후의 분류 성능을 비교합니다."
)

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
# 입력 속성 선택: 분류 모델 페이지와 동일한 기본 선택
# ------------------------------------------------------------
st.subheader("1. 모델에 사용할 속성 고르기")
selected_features = st.multiselect(
    "입력 속성 (두 개 이상 선택)",
    options=list(FEATURE_LABELS),
    default=["age", "avg_glucose_level", "hypertension", "heart_disease"],
    format_func=lambda feature: FEATURE_LABELS[feature],
    help="체질량지수(bmi)를 선택하면 학습용 중앙값으로 빈 값을 채웁니다.",
)

if len(selected_features) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요.")
    st.stop()


# ------------------------------------------------------------
# 채점표 페이지와 같은 데이터 분할
# ------------------------------------------------------------
df = df.sort_values("id", kind="mergesort").reset_index(drop=True)
test_mask = [position % 10 < 3 for position in range(len(df))]
train_df = df.loc[[not value for value in test_mask]].copy()
test_df = df.loc[test_mask].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

st.markdown(
    f"**전체 {len(df):,}명**　|　학습용 **{len(train_df):,}명**　|　"
    f"테스트용 **{len(test_df):,}명**　|　"
    f"테스트용 실제 뇌졸중 **{int(y_test.eq(1).sum()):,}명**"
)
if len(test_df) != 1533:
    st.warning(
        f"현재 테스트용 인원은 {len(test_df):,}명입니다. "
        "원본 데이터의 행 수를 확인해 주세요."
    )

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

# BMI는 사용자가 선택한 경우에만 학습용 중앙값으로 대체
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
# 크기 맞추기: 학습용 데이터에만 fit
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# 전·후 모델 학습
# 전: class_weight=None / 후: class_weight="balanced"
# ------------------------------------------------------------
logistic_before = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
)
logistic_after = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)

tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)

logistic_before.fit(X_train_scaled, y_train)
logistic_after.fit(X_train_scaled, y_train)
tree_before.fit(X_train, y_train)
tree_after.fit(X_train, y_train)

models = {
    LOGISTIC_NAME: {
        "before": logistic_before,
        "after": logistic_after,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
    },
    TREE_NAME: {
        "before": tree_before,
        "after": tree_after,
        "X_train": X_train,
        "X_test": X_test,
    },
}


# ------------------------------------------------------------
# 혼동행렬과 지표 계산
# ------------------------------------------------------------
def get_confusion(actual, predicted):
    actual = pd.Series(actual).reset_index(drop=True).astype(int)
    predicted = pd.Series(predicted).reset_index(drop=True).astype(int)
    return {
        "TP": int(((actual == 1) & (predicted == 1)).sum()),
        "FP": int(((actual == 0) & (predicted == 1)).sum()),
        "FN": int(((actual == 1) & (predicted == 0)).sum()),
        "TN": int(((actual == 0) & (predicted == 0)).sum()),
    }


def get_metrics(counts, total):
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) else None

    if tp + fp == 0:
        precision = None
        f1 = None
    else:
        precision = tp / (tp + fp)
        f1 = (
            2 * precision * recall / (precision + recall)
            if recall is not None and precision + recall > 0
            else None
        )

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1,
    }


def metric_text(value):
    return "계산할 수 없음" if value is None else f"{value * 100:.2f}%"


results = {}
for model_name, model_info in models.items():
    results[model_name] = {}
    for stage in ["before", "after"]:
        estimator = model_info[stage]
        predictions = estimator.predict(model_info["X_test"])
        counts = get_confusion(y_test, predictions)
        metrics = get_metrics(counts, len(y_test))
        results[model_name][stage] = {
            "predictions": predictions,
            "counts": counts,
            "metrics": metrics,
        }


# ------------------------------------------------------------
# 모델별 전·후 혼동행렬 4칸 + 지표 표
# ------------------------------------------------------------
st.divider()
st.subheader("2. 가중치를 맞추기 전과 후의 채점 결과")
st.write(
    "가중치를 맞춘 뒤에는 뇌졸중(1)과 뇌졸중 아님(0)을 학습할 때 "
    "각 클래스의 중요도를 다르게 반영합니다. 테스트 데이터는 전·후 모두 같습니다."
)

for model_name in [LOGISTIC_NAME, TREE_NAME]:
    st.markdown(f"### {model_name}")
    before = results[model_name]["before"]
    after = results[model_name]["after"]

    def evaluation_table(result):
        c = result["counts"]
        m = result["metrics"]
        return pd.DataFrame(
            [
                {"항목": "TP (실제 뇌졸중·예측 뇌졸중)", "값": f"{c['TP']:,}명"},
                {"항목": "FP (실제 아님·예측 뇌졸중)", "값": f"{c['FP']:,}명"},
                {"항목": "FN (실제 뇌졸중·예측 아님)", "값": f"{c['FN']:,}명"},
                {"항목": "TN (실제 아님·예측 아님)", "값": f"{c['TN']:,}명"},
                {"항목": "정확도", "값": metric_text(m["정확도"])},
                {"항목": "재현율", "값": metric_text(m["재현율"])},
                {"항목": "정밀도", "값": metric_text(m["정밀도"])},
                {"항목": "F1", "값": metric_text(m["F1"])},
            ]
        )

    before_table = evaluation_table(before).rename(columns={"값": "가중치 적용 전"})
    after_table = evaluation_table(after).rename(columns={"값": "가중치 적용 후"})
    comparison_table = before_table.merge(
        after_table,
        on="항목",
        how="inner",
    )
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)

    b = before["counts"]
    a = after["counts"]
    st.markdown(
        f"**변화:** 찾아낸 환자(TP) {b['TP']:,}명 → {a['TP']:,}명　|　"
        f"놓친 환자(FN) {b['FN']:,}명 → {a['FN']:,}명　|　"
        f"헛짚은 사람(FP) {b['FP']:,}명 → {a['FP']:,}명"
    )
    st.divider()


# ------------------------------------------------------------
# 산점도: 선택 속성 수에 따라 2D 또는 3D
# 경계는 로지스틱 회귀의 확률 0.5
# ------------------------------------------------------------
st.subheader("3. 가중치 적용 전·후의 확률 경계 비교")
st.write(
    "점의 색은 테스트 데이터의 실제 뇌졸중 여부를 나타냅니다. "
    "두 경계는 로지스틱 회귀가 뇌졸중일 확률을 0.5로 판단하는 위치입니다."
)

if len(selected_features) == 2:
    x_feature, y_feature = selected_features[0], selected_features[1]
    x_values = X_test[x_feature]
    y_values = X_test[y_feature]

    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_pad = (x_max - x_min) * 0.05 if x_max != x_min else 0.5
    y_pad = (y_max - y_min) * 0.05 if y_max != y_min else 0.5

    grid_size = 100
    x_grid = [
        x_min - x_pad + (x_max - x_min + 2 * x_pad) * i / (grid_size - 1)
        for i in range(grid_size)
    ]
    y_grid = [
        y_min - y_pad + (y_max - y_min + 2 * y_pad) * i / (grid_size - 1)
        for i in range(grid_size)
    ]

    grid = pd.MultiIndex.from_product(
        [y_grid, x_grid],
        names=[y_feature, x_feature],
    ).to_frame(index=False)
    for feature in selected_features:
        if feature not in [x_feature, y_feature]:
            grid[feature] = float(X_test[feature].median())
    grid = grid[selected_features]

    fig = go.Figure()
    colors = {"before": "#E07A2D", "after": "#7A4CC2"}
    boundary_status = {}

    for stage, label in [
        ("before", "가중치 적용 전 0.5 경계"),
        ("after", "가중치 적용 후 0.5 경계"),
    ]:
        model = models[LOGISTIC_NAME][stage]
        probabilities = model.predict_proba(scaler.transform(grid))[:, 1]
        z_matrix = probabilities.reshape(grid_size, grid_size)
        p_min, p_max = float(probabilities.min()), float(probabilities.max())
        boundary_status[stage] = (p_min <= 0.5 <= p_max, p_min, p_max)

        if boundary_status[stage][0]:
            fig.add_trace(
                go.Contour(
                    x=x_grid,
                    y=y_grid,
                    z=z_matrix,
                    contours={
                        "start": 0.5,
                        "end": 0.5,
                        "size": 1,
                        "coloring": "none",
                        "showlabels": False,
                    },
                    line={"color": colors[stage], "width": 3, "dash": "solid"},
                    showscale=False,
                    hoverinfo="skip",
                    name=label,
                )
            )

    for actual, label, color in [
        (0, "실제 뇌졸중 아님(0)", "#2878B5"),
        (1, "실제 뇌졸중(1)", "#D64B4B"),
    ]:
        points = test_df.loc[y_test.eq(actual)]
        fig.add_trace(
            go.Scatter(
                x=points[x_feature],
                y=points[y_feature],
                mode="markers",
                name=label,
                marker={"color": color, "size": 6, "opacity": 0.65},
            )
        )

    fig.update_layout(
        title="테스트 데이터와 로지스틱 회귀의 확률 경계",
        xaxis_title=FEATURE_LABELS[x_feature],
        yaxis_title=FEATURE_LABELS[y_feature],
        height=650,
        legend_title="범례",
    )
    st.plotly_chart(fig, use_container_width=True)

    for stage, label in [
        ("before", "가중치 적용 전"),
        ("after", "가중치 적용 후"),
    ]:
        in_view, p_min, p_max = boundary_status[stage]
        if not in_view:
            st.info(
                f"{label} 경계선은 현재 그림 범위 밖에 있습니다 "
                f"(그림 영역에서 예측 확률 {p_min:.3f}~{p_max:.3f})."
            )
    fixed_features = [
        feature for feature in selected_features
        if feature not in [x_feature, y_feature]
    ]
    if fixed_features:
        fixed_text = ", ".join(
            f"{FEATURE_LABELS[feature]} {float(X_test[feature].median()):.2f}"
            for feature in fixed_features
        )
        st.caption(f"두 축이 아닌 속성은 테스트 데이터 중앙값으로 고정했습니다: {fixed_text}")

else:
    # 3개 이상 선택하면 첫 세 속성을 3차원 축으로 사용
    x_feature, y_feature, z_feature = selected_features[:3]
    st.caption(
        f"3차원 축: 가로 {FEATURE_LABELS[x_feature]}, "
        f"세로 {FEATURE_LABELS[y_feature]}, 높이 {FEATURE_LABELS[z_feature]}. "
        "선택한 나머지 속성은 테스트 데이터 중앙값으로 고정합니다."
    )

    x_values = X_test[x_feature]
    y_values = X_test[y_feature]
    z_values = X_test[z_feature]
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    z_min, z_max = float(z_values.min()), float(z_values.max())

    def padded_range(low, high):
        padding = (high - low) * 0.05 if high != low else 0.5
        return low - padding, high + padding

    x_low, x_high = padded_range(x_min, x_max)
    y_low, y_high = padded_range(y_min, y_max)
    z_low, z_high = padded_range(z_min, z_max)

    mesh_count = 35
    x_mesh = [
        x_low + (x_high - x_low) * i / (mesh_count - 1)
        for i in range(mesh_count)
    ]
    y_mesh = [
        y_low + (y_high - y_low) * i / (mesh_count - 1)
        for i in range(mesh_count)
    ]

    fig = go.Figure()
    boundary_status_3d = {}

    for stage, label, color in [
        ("before", "가중치 적용 전 0.5 평면", "#E07A2D"),
        ("after", "가중치 적용 후 0.5 평면", "#7A4CC2"),
    ]:
        model = models[LOGISTIC_NAME][stage]
        coefficients = model.coef_[0]
        intercept = float(model.intercept_[0])
        feature_positions = {feature: i for i, feature in enumerate(selected_features)}

        x_pos = feature_positions[x_feature]
        y_pos = feature_positions[y_feature]
        z_pos = feature_positions[z_feature]

        # StandardScaler 좌표계에서 확률 0.5는 선형식의 값이 0인 평면
        means = scaler.mean_
        scales = scaler.scale_
        coef_original = coefficients / scales
        intercept_original = intercept - float((coefficients * means / scales).sum())

        # 세 번째 축 계수가 0에 가까우면 이 축 방향으로 평면을 표현할 수 없음
        if abs(coef_original[z_pos]) < 1e-12:
            boundary_status_3d[stage] = "세로축 방향으로 평면을 계산할 수 없음"
            continue

        z_plane = []
        for y_value in y_mesh:
            row = []
            for x_value in x_mesh:
                linear_without_z = intercept_original
                linear_without_z += coef_original[x_pos] * x_value
                linear_without_z += coef_original[y_pos] * y_value

                for feature in selected_features[3:]:
                    fixed_value = float(X_test[feature].median())
                    linear_without_z += (
                        coef_original[feature_positions[feature]] * fixed_value
                    )

                z_value = -linear_without_z / coef_original[z_pos]
                row.append(z_value)
            z_plane.append(row)

        finite_z = [
            value for row in z_plane for value in row
            if pd.notna(value) and abs(value) != float("inf")
        ]
        if finite_z and min(finite_z) <= z_high and max(finite_z) >= z_low:
            boundary_status_3d[stage] = "그림 범위와 만남"
            fig.add_trace(
                go.Surface(
                    x=x_mesh,
                    y=y_mesh,
                    z=z_plane,
                    opacity=0.38,
                    showscale=False,
                    colorscale=[[0, color], [1, color]],
                    name=label,
                    hovertemplate=(
                        f"{label}<br>{FEATURE_LABELS[x_feature]}: %{{x:.2f}}"
                        f"<br>{FEATURE_LABELS[y_feature]}: %{{y:.2f}}"
                        f"<br>{FEATURE_LABELS[z_feature]}: %{{z:.2f}}<extra></extra>"
                    ),
                )
            )
        else:
            boundary_status_3d[stage] = "평면이 현재 그림 범위 밖에 있음"

    for actual, label, color in [
        (0, "실제 뇌졸중 아님(0)", "#2878B5"),
        (1, "실제 뇌졸중(1)", "#D64B4B"),
    ]:
        points = test_df.loc[y_test.eq(actual)]
        fig.add_trace(
            go.Scatter3d(
                x=points[x_feature],
                y=points[y_feature],
                z=points[z_feature],
                mode="markers",
                name=label,
                marker={"color": color, "size": 3, "opacity": 0.7},
            )
        )

    fig.update_layout(
        title="테스트 데이터와 로지스틱 회귀의 0.5 평면",
        scene={
            "xaxis_title": FEATURE_LABELS[x_feature],
            "yaxis_title": FEATURE_LABELS[y_feature],
            "zaxis_title": FEATURE_LABELS[z_feature],
        },
        height=750,
        legend_title="범례",
    )
    st.plotly_chart(fig, use_container_width=True)

    for stage, label in [
        ("before", "가중치 적용 전"),
        ("after", "가중치 적용 후"),
    ]:
        st.write(f"**{label}:** {boundary_status_3d.get(stage, '평면을 표시하지 못했습니다.')}")
    if len(selected_features) > 3:
        fixed_text = ", ".join(
            f"{FEATURE_LABELS[feature]} {float(X_test[feature].median()):.2f}"
            for feature in selected_features[3:]
        )
        st.caption(f"세 축이 아닌 속성은 테스트 데이터 중앙값으로 고정했습니다: {fixed_text}")

st.caption(
    "※ 가중치 적용은 학습 과정에서 소수 클래스의 오분류에 더 큰 중요도를 주는 방법입니다. "
    "재현율이 높아질 수 있지만 헛짚은 사람도 늘어날 수 있습니다. "
    "이 결과는 학습용 실습이며 실제 의료 진단을 대신하지 않습니다."
)
