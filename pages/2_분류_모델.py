import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree


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

st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🧪",
    layout="wide",
)


# ------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🧪 분류 모델")
st.caption(
    "정해진 방법으로 학습용 데이터와 테스트용 데이터를 나누고, "
    "세 가지 분류 방법의 성능과 판단 과정을 비교합니다."
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

# ID 오름차순 정렬 후 10명 단위 묶음의 앞 3명을 테스트용으로 고정합니다.
df = df.sort_values("id", kind="mergesort").reset_index(drop=True)
test_mask = [position % 10 < 3 for position in range(len(df))]
train_df = df.loc[[not value for value in test_mask]].copy()
test_df = df.loc[test_mask].copy()

st.subheader("데이터 나누기")
split_col1, split_col2, split_col3 = st.columns(3)
split_col1.metric("전체 사람 수", f"{len(df):,}명")
split_col2.metric("학습용 사람 수", f"{len(train_df):,}명")
split_col3.metric("테스트용 사람 수", f"{len(test_df):,}명")
st.caption(
    "번호(id) 순으로 정렬한 뒤 10명씩 묶어 각 묶음의 앞 3명을 테스트용으로, "
    "나머지 7명을 학습용으로 사용합니다. BMI 대체값과 크기 맞추기 기준도 "
    "학습용 데이터에서만 계산합니다."
)
if len(test_df) != 1533:
    st.warning(
        f"현재 데이터의 테스트용 인원은 {len(test_df):,}명입니다. "
        "요청한 1,533명과 다르다면 원본 데이터의 행 수를 확인해 주세요."
    )


# ------------------------------------------------------------
# 입력 속성 선택
# ------------------------------------------------------------
st.divider()
st.subheader("1. 모델에 사용할 속성 고르기")
feature_options = list(FEATURE_LABELS)
default_features = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

selected_features = st.multiselect(
    "입력 속성 (두 개 이상 선택)",
    options=feature_options,
    default=default_features,
    format_func=lambda feature: FEATURE_LABELS[feature],
    help="처음에는 체질량지수(bmi)를 제외한 네 가지 속성이 선택되어 있습니다.",
)

if len(selected_features) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요.")
    st.stop()


# ------------------------------------------------------------
# 입력 데이터 준비: BMI 결측값은 선택했을 때만 학습용 중앙값으로 대체
# ------------------------------------------------------------
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()
y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

imputation_notes = []
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()
    if pd.isna(bmi_median):
        st.error("학습용 데이터에서 체질량지수(bmi)의 중앙값을 계산할 수 없습니다.")
        st.stop()

    train_bmi_missing = int(X_train["bmi"].isna().sum())
    test_bmi_missing = int(X_test["bmi"].isna().sum())
    X_train.loc[:, "bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test.loc[:, "bmi"] = X_test["bmi"].fillna(bmi_median)
    imputation_notes.append(
        f"체질량지수의 빈 값은 학습용 데이터의 중앙값 {bmi_median:.2f}으로 "
        f"채웠습니다 (학습용 {train_bmi_missing:,}명, 테스트용 {test_bmi_missing:,}명)."
    )
else:
    imputation_notes.append(
        "체질량지수(bmi)를 선택하지 않아 BMI 빈 값은 처리하지 않았습니다."
    )

# 선택한 입력 속성에 결측값이 더 있는지 확인합니다.
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

for note in imputation_notes:
    st.caption(note)


# ------------------------------------------------------------
# 모델 학습
# ------------------------------------------------------------
# 크기 맞추기는 학습용 데이터로만 학습합니다.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 로지스틱 회귀: 확률 0.5를 기준으로 0/1 분류
logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
)
logistic_model.fit(X_train_scaled, y_train)

# 의사결정트리: 최대 질문 3회, 마지막 마디 최소 인원 5명
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
tree_model.fit(X_train, y_train)

# 아무 입력도 보지 않고 학습용에서 더 많은 쪽으로만 답하는 기준 모델
majority_class = int(y_train.value_counts().idxmax())
baseline_train_pred = [majority_class] * len(y_train)
baseline_test_pred = [majority_class] * len(y_test)

logistic_train_accuracy = accuracy_score(
    y_train, logistic_model.predict(X_train_scaled)
)
logistic_test_accuracy = accuracy_score(
    y_test, logistic_model.predict(X_test_scaled)
)
tree_train_accuracy = accuracy_score(y_train, tree_model.predict(X_train))
tree_test_accuracy = accuracy_score(y_test, tree_model.predict(X_test))
baseline_train_accuracy = accuracy_score(y_train, baseline_train_pred)
baseline_test_accuracy = accuracy_score(y_test, baseline_test_pred)


# ------------------------------------------------------------
# 정확도 카드
# ------------------------------------------------------------
st.divider()
st.subheader("2. 모델 정확도 비교")
st.write(
    "큰 숫자는 테스트 데이터의 정확도입니다. 아래에는 학습 데이터와 "
    "테스트 데이터의 정확도를 나란히 표시했습니다."
)


def show_accuracy_card(container, title, train_accuracy, test_accuracy):
    with container:
        st.markdown(
            f"""
            <div style="
                border: 1px solid rgba(128,128,128,0.35);
                border-radius: 12px;
                padding: 18px 18px 14px 18px;
                min-height: 150px;
            ">
                <div style="font-size: 1rem; font-weight: 650; margin-bottom: 10px;">
                    {title}
                </div>
                <div style="font-size: 2.35rem; font-weight: 750; line-height: 1.2;">
                    {test_accuracy * 100:.2f}%
                </div>
                <div style="font-size: 0.82rem; color: gray; margin-top: 12px;">
                    학습 데이터 {train_accuracy * 100:.2f}%
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    테스트 데이터 {test_accuracy * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


accuracy_cols = st.columns(3)
show_accuracy_card(
    accuracy_cols[0],
    "로지스틱 회귀(확률로 답하는 모델)",
    logistic_train_accuracy,
    logistic_test_accuracy,
)
show_accuracy_card(
    accuracy_cols[1],
    "의사결정트리(질문으로 답하는 모델)",
    tree_train_accuracy,
    tree_test_accuracy,
)
show_accuracy_card(
    accuracy_cols[2],
    "입력을 보지 않는 다수결 모델",
    baseline_train_accuracy,
    baseline_test_accuracy,
)
st.caption(
    f"다수결 모델은 모든 사람에게 학습용에서 더 많았던 "
    f"'{ '뇌졸중' if majority_class == 1 else '아님' }'으로 답합니다. "
    "정확도는 전체 정답 중 맞힌 답의 비율입니다."
)


# ------------------------------------------------------------
# 2개 속성을 고른 2차원 그림
# ------------------------------------------------------------
st.divider()
st.subheader("3. 두 속성으로 보는 분류 결과")
st.write(
    "가로축과 세로축으로 사용할 속성을 고르세요. 점의 색은 테스트 데이터의 "
    "실제 뇌졸중 여부이고, 배경은 의사결정트리가 나눈 영역입니다. "
    "선은 로지스틱 회귀 모델의 예측 확률이 0.5인 경계입니다."
)

axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_feature = st.selectbox(
        "가로축",
        options=selected_features,
        index=0,
        format_func=lambda feature: FEATURE_LABELS[feature],
        key="x_axis_feature",
    )
with axis_col2:
    y_default_index = 1 if selected_features[1] != x_feature else 0
    y_options = [feature for feature in selected_features if feature != x_feature]
    y_feature = st.selectbox(
        "세로축",
        options=y_options,
        index=0,
        format_func=lambda feature: FEATURE_LABELS[feature],
        key="y_axis_feature",
    )

# 두 축 이외의 속성은 테스트 데이터 중앙값으로 고정합니다.
fixed_values = {}
for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = float(X_test[feature].median())

if fixed_values:
    fixed_text = ", ".join(
        f"{FEATURE_LABELS[feature]} = {value:.2f}"
        for feature, value in fixed_values.items()
    )
    st.write(f"**두 축이 아닌 속성을 고정한 값(테스트 데이터 중앙값):** {fixed_text}")
else:
    st.write("선택한 속성이 두 개이므로 고정한 나머지 속성은 없습니다.")

# 테스트 데이터 범위 안팎에 약간의 여백을 둔 격자 생성
grid_size = 75
x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())
y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

x_padding = (x_max - x_min) * 0.04 if x_max != x_min else 0.5
y_padding = (y_max - y_min) * 0.04 if y_max != y_min else 0.5

x_grid = [
    x_min - x_padding
    + (x_max - x_min + 2 * x_padding) * index / (grid_size - 1)
    for index in range(grid_size)
]
y_grid = [
    y_min - y_padding
    + (y_max - y_min + 2 * y_padding) * index / (grid_size - 1)
    for index in range(grid_size)
]

# y가 행, x가 열이 되도록 격자 데이터 구성
grid_df = pd.MultiIndex.from_product(
    [y_grid, x_grid],
    names=[y_feature, x_feature],
).to_frame(index=False)

for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        grid_df[feature] = fixed_values[feature]

grid_df = grid_df[selected_features]

# 로지스틱 회귀의 확률 경계 계산
grid_scaled = scaler.transform(grid_df)
grid_probabilities = logistic_model.predict_proba(grid_scaled)[:, 1]
probability_matrix = grid_probabilities.reshape(grid_size, grid_size)
probability_min = float(grid_probabilities.min())
probability_max = float(grid_probabilities.max())
boundary_in_view = probability_min <= 0.5 <= probability_max

# 의사결정트리의 분류 영역 계산
tree_grid_predictions = tree_model.predict(grid_df)
tree_region_matrix = tree_grid_predictions.reshape(grid_size, grid_size)

figure = go.Figure()

# 의사결정트리 영역을 옅은 색으로 표시
figure.add_trace(
    go.Heatmap(
        x=x_grid,
        y=y_grid,
        z=tree_region_matrix,
        zmin=0,
        zmax=1,
        colorscale=[
            [0.0, "rgba(80, 150, 220, 0.18)"],
            [0.499, "rgba(80, 150, 220, 0.18)"],
            [0.5, "rgba(230, 100, 100, 0.18)"],
            [1.0, "rgba(230, 100, 100, 0.18)"],
        ],
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 영역",
    )
)

# 로지스틱 회귀 확률 0.5 경계선
if boundary_in_view:
    figure.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=probability_matrix,
            contours={
                "start": 0.5,
                "end": 0.5,
                "size": 1,
                "coloring": "none",
                "showlabels": False,
            },
            line={"color": "#222222", "width": 3},
            showscale=False,
            hoverinfo="skip",
            name="로지스틱 회귀 경계(0.5)",
        )
    )

# 테스트 데이터의 실제 정답을 색으로 구분
test_plot = test_df[[x_feature, y_feature, "stroke"]].copy()
test_plot["실제 결과"] = test_plot["stroke"].map(
    {0: "뇌졸중 아님(0)", 1: "뇌졸중(1)"}
)

for actual_value, label, color in [
    (0, "뇌졸중 아님(0)", "#2878B5"),
    (1, "뇌졸중(1)", "#D64B4B"),
]:
    points = test_plot.loc[test_plot["stroke"] == actual_value]
    figure.add_trace(
        go.Scatter(
            x=points[x_feature],
            y=points[y_feature],
            mode="markers",
            name=label,
            marker={
                "color": color,
                "size": 6,
                "opacity": 0.68,
                "line": {"width": 0.4, "color": "white"},
            },
            hovertemplate=(
                f"{FEATURE_LABELS[x_feature]}: %{{x:.2f}}<br>"
                f"{FEATURE_LABELS[y_feature]}: %{{y:.2f}}<br>"
                f"실제 결과: {label}<extra></extra>"
            ),
        )
    )

figure.update_layout(
    title="테스트 데이터와 두 모델의 분류 경계",
    xaxis_title=FEATURE_LABELS[x_feature],
    yaxis_title=FEATURE_LABELS[y_feature],
    legend_title="테스트 데이터의 실제 결과",
    height=650,
)
st.plotly_chart(figure, use_container_width=True)

if not boundary_in_view:
    st.info(
        "로지스틱 회귀의 확률 0.5 경계선이 현재 그림 범위 안에 없습니다. "
        f"현재 격자에서 계산된 확률 범위는 {probability_min:.3f}~"
        f"{probability_max:.3f}입니다."
    )
else:
    st.caption("검은 선이 로지스틱 회귀 모델의 예측 확률 0.5 경계입니다.")

st.caption(
    "배경의 파란색·붉은색 영역은 의사결정트리가 각각 0(뇌졸중 아님), "
    "1(뇌졸중)로 분류한 영역입니다. 다른 속성은 위에 적힌 중앙값으로 고정해 계산했습니다."
)


# ------------------------------------------------------------
# 의사결정트리 질문 구조(DOT)
# ------------------------------------------------------------
st.divider()
st.subheader("4. 의사결정트리가 던진 질문")
st.write(
    "의사결정트리는 학습용 데이터에 질문을 차례로 적용합니다. "
    "왼쪽 가지는 조건에 '예(≤)'인 경우, 오른쪽 가지는 '아니요(>)'인 경우입니다."
)

tree_structure = tree_model.tree_
decision_path = tree_model.decision_path(X_train)
leaf_nodes = []
negative_leaf_nodes = []
asked_feature_indices = []

for node_index in range(tree_structure.node_count):
    left_child = tree_structure.children_left[node_index]
    right_child = tree_structure.children_right[node_index]
    is_leaf = left_child == _tree.TREE_LEAF and right_child == _tree.TREE_LEAF

    # 해당 마디에 도달한 학습용 행을 decision_path로 확인
    reached_rows = decision_path[:, node_index].toarray().ravel().astype(bool)
    node_people = int(reached_rows.sum())
    node_strokes = int(y_train.iloc[reached_rows].eq(1).sum())
    node_rate = node_strokes / node_people * 100 if node_people else 0.0

    if is_leaf:
        leaf_nodes.append(node_index)
        leaf_class_index = int(tree_structure.value[node_index][0].argmax())
        leaf_prediction = int(tree_model.classes_[leaf_class_index])
        if leaf_prediction == 0:
            negative_leaf_nodes.append(node_index)

    if not is_leaf and tree_structure.feature[node_index] not in asked_feature_indices:
        asked_feature_indices.append(int(tree_structure.feature[node_index]))

    if is_leaf:
        leaf_class_index = int(tree_structure.value[node_index][0].argmax())
        leaf_prediction = int(tree_model.classes_[leaf_class_index])
        answer_text = "답: 뇌졸중(1)" if leaf_prediction == 1 else "답: 뇌졸중 아님(0)"
        fill_color = "#F8CACA" if leaf_prediction == 1 else "#C9E2F5"
        node_label = (
            f"{answer_text}\\n"
            f"훈련 인원: {node_people}명\\n"
            f"실제 뇌졸중: {node_strokes}명 ({node_rate:.1f}%)"
        )
    else:
        split_feature = selected_features[int(tree_structure.feature[node_index])]
        threshold = float(tree_structure.threshold[node_index])
        node_label = (
            f"질문: {FEATURE_LABELS[split_feature]} ≤ {threshold:.2f}?\\n"
            f"훈련 인원: {node_people}명\\n"
            f"실제 뇌졸중: {node_strokes}명 ({node_rate:.1f}%)"
        )
        fill_color = "#FFF0C2"

    # DOT 문자열의 따옴표와 줄바꿈을 안전하게 처리
    escaped_label = node_label.replace('"', r'\"')
    if node_index == 0:
        dot_nodes = []
    dot_nodes.append(
        f'  n{node_index} [label="{escaped_label}", fillcolor="{fill_color}"];'
    )

dot_edges = []
for node_index in range(tree_structure.node_count):
    left_child = tree_structure.children_left[node_index]
    right_child = tree_structure.children_right[node_index]
    if left_child != _tree.TREE_LEAF:
        split_feature = selected_features[int(tree_structure.feature[node_index])]
        threshold = float(tree_structure.threshold[node_index])
        dot_edges.append(
            f'  n{node_index} -> n{left_child} '
            f'[label="예: {FEATURE_LABELS[split_feature]} ≤ {threshold:.2f}"];'
        )
        dot_edges.append(
            f'  n{node_index} -> n{right_child} '
            f'[label="아니요: {FEATURE_LABELS[split_feature]} > {threshold:.2f}"];'
        )

dot_source = "\n".join(
    [
        "digraph DecisionTree {",
        '  graph [rankdir=TB, bgcolor="transparent", nodesep=0.35, ranksep=0.55];',
        '  node [shape=box, style="rounded,filled", color="#667085", fontname="sans-serif", fontsize=10, margin="0.14,0.10"];',
        '  edge [color="#667085", fontname="sans-serif", fontsize=9, arrowsize=0.7];',
    ]
    + dot_nodes
    + dot_edges
    + ["}"]
)

st.graphviz_chart(dot_source, use_container_width=True)

asked_features = [
    selected_features[index] for index in asked_feature_indices
]
asked_feature_names = [FEATURE_LABELS[feature] for feature in asked_features]

summary_col1, summary_col2 = st.columns(2)
summary_col1.write(
    f"**답을 내는 마디(잎)는 모두 {len(leaf_nodes)}칸이며, "
    f"그중 {len(negative_leaf_nodes)}칸이 '뇌졸중 아님(0)'이라고 답합니다.**"
)
summary_col2.write(
    "**이 나무가 실제로 물은 속성:** "
    + (", ".join(asked_feature_names) if asked_feature_names else "질문 없음")
)

st.caption(
    "마디의 인원과 뇌졸중 비율은 해당 마디에 도달한 학습용 데이터만으로 계산했습니다. "
    "나뭇잎 색은 모델이 내리는 답을 나타냅니다."
)
st.caption("※ 이 분류 모델은 학습용 실습 도구이며 실제 의료 진단이나 치료 결정을 대신하지 않습니다.")
