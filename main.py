import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="마르코프 체인 DNA 탐구", page_icon="🧬", layout="wide")


# ---------- 계산 함수 ----------
def simulate(p_ag, p_ga, a0, n):
    """A 빈도 a0에서 시작해 n세대까지 (A, G) 빈도를 반환"""
    M = np.array([[1 - p_ag, p_ag], [p_ga, 1 - p_ga]])
    v = np.array([a0, 1 - a0])
    rows = [(0, v[0], v[1])]
    for i in range(1, n + 1):
        v = M.T @ v
        rows.append((i, v[0], v[1]))
    return pd.DataFrame(rows, columns=["세대", "A 빈도", "G 빈도"])


def stationary(p_ag, p_ga):
    """정상 분포 (A, G). p_ag + p_ga = 0이면 유일하지 않음"""
    s = p_ag + p_ga
    if s == 0:
        return None
    return p_ga / s, p_ag / s


def line_chart(dfs, labels, st_a=None):
    fig = go.Figure()
    for df, lab in zip(dfs, labels):
        fig.add_trace(go.Scatter(x=df["세대"], y=df["A 빈도"] * 100,
                                 mode="lines+markers", name=lab))
    if st_a is not None:
        fig.add_hline(y=st_a * 100, line_dash="dash", line_color="gray",
                      annotation_text=f"정상 분포 A = {st_a*100:.1f}%")
    fig.update_layout(xaxis_title="세대", yaxis_title="A 빈도 (%)",
                      yaxis_range=[0, 100], height=420)
    return fig


# ---------- 사이드바 ----------
st.sidebar.title("🧬 탐구 목차")
page = st.sidebar.radio("이동", [
    "1. 탐구 동기와 핵심 질문",
    "2. 관련 개념",
    "3. 세대별 빈도 계산",
    "4. 정상 분포 구하기",
    "5. 초기 빈도 비교",
    "6. 전이 확률 비교",
    "7. 결과 해석",
    "8. 생명과학·약학 연결",
    "🧪 직접 해보는 시뮬레이터",
])

# ---------- 1 ----------
if page.startswith("1"):
    st.title("마르코프 체인의 전이 확률 변화에 따른 DNA 염기 빈도의 장기적 수렴 분석")
    st.header("1. 탐구 동기와 핵심 질문")
    st.write("""
DNA는 A(아데닌), T(티민), G(구아닌), C(사이토신) 네 염기로 구성됩니다.
DNA 복제 과정 등에서 돌연변이가 일어나면 특정 염기가 다른 염기로 바뀔 수 있습니다.
이 변화를 **마르코프 체인**(현재 상태와 전이 확률로 다음 상태를 예측하는 확률 모델)으로 분석합니다.
""")
    st.subheader("핵심 질문")
    st.markdown("""
1. 초기 염기 빈도가 달라도 장기적으로 같은 빈도에 수렴할까?
2. 염기 사이의 전이 확률이 달라지면 수렴값은 어떻게 바뀔까?
3. 이러한 결과는 어떤 수학적 원리로 설명할 수 있을까?
""")
    st.info("**탐구 목적**: 초기 조건과 전이 확률이 염기 빈도의 장기적 변화에 미치는 영향을 수학적으로 분석한다.")

# ---------- 2 ----------
elif page.startswith("2"):
    st.header("2. 관련 개념 이해하기")

    st.subheader("2-1. 마르코프 체인")
    st.write("""
계산을 간단히 하기 위해 **A와 G 두 염기만** 고려합니다.
현재 상태만 알면 다음 상태의 확률을 계산할 수 있다고 가정하는 것이 마르코프 체인의 핵심입니다.
""")

    st.subheader("2-2. 전이 확률 행렬")
    st.table(pd.DataFrame({"다음 세대 A": [0.8, 0.3], "다음 세대 G": [0.2, 0.7]},
                          index=["현재 A", "현재 G"]))
    st.latex(r"M=\begin{pmatrix}0.8&0.2\\0.3&0.7\end{pmatrix}")
    st.write("각 행의 합은 1입니다. 현재 A인 염기는 다음 세대에 A 또는 G가 되기 때문입니다.")
    st.warning("이 행렬은 실제 돌연변이율 측정값이 아니라, 수학적 원리를 분석하기 위한 **가상의 확률**입니다.")

    st.subheader("2-3. 염기 빈도 벡터")
    st.latex(r"v_0=\begin{pmatrix}0.9\\0.1\end{pmatrix}")
    st.latex(r"v_1 = M^{T} v_0")
    st.write("빈도 벡터를 열벡터로 정의했기 때문에 전치행렬 $M^T$를 곱합니다.")

# ---------- 3 ----------
elif page.startswith("3"):
    st.header("3. 탐구 과정 ①: 세대가 반복될 때 염기 빈도 계산")
    st.latex(r"v_{n+1}=M^{T}v_n")
    st.write("초기 빈도: A 90%, G 10%")
    st.latex(r"""v_1=\begin{pmatrix}0.8&0.3\\0.2&0.7\end{pmatrix}
\begin{pmatrix}0.9\\0.1\end{pmatrix}""")
    st.latex(r"A:\ 0.8\times0.9+0.3\times0.1=0.75 \qquad G:\ 0.2\times0.9+0.7\times0.1=0.25")

    n = st.slider("계산할 세대 수", 5, 30, 5)
    df = simulate(0.2, 0.3, 0.9, n)
    show = df.copy()
    show["A 빈도"] = (show["A 빈도"] * 100).round(3).astype(str) + "%"
    show["G 빈도"] = (show["G 빈도"] * 100).round(3).astype(str) + "%"
    c1, c2 = st.columns([1, 2])
    c1.dataframe(show, hide_index=True, use_container_width=True)
    c2.plotly_chart(line_chart([df], ["A 빈도"], 0.6), use_container_width=True)

    st.success("""
**발견 1**: 염기 빈도가 일정한 값에 가까워지는 경향이 있고, 변화량은 점점 작아집니다.

그렇다면 **왜 하필 A는 60%, G는 40%에 가까워질까?** → 다음 단계에서 분석합니다.
""")

# ---------- 4 ----------
elif page.startswith("4"):
    st.header("4. 탐구 과정 ②: 장기적으로 수렴하는 값 찾기")
    st.write("A의 장기 빈도를 $a$라 하면, 다음 세대에도 $a$여야 합니다. ($1-a$는 G의 빈도)")
    st.latex(r"a=0.8a+0.3(1-a)")
    st.latex(r"a=0.5a+0.3 \;\Rightarrow\; 0.5a=0.3 \;\Rightarrow\; \boxed{a=0.6}")
    st.write("""
이 값을 **정상 분포(stationary distribution)** 라고 합니다.
전이 확률을 한 번 더 적용해도 분포가 바뀌지 않는 상태입니다.
""")
    st.warning("정상 분포는 실제 생물 집단에서 관측되는 빈도가 아니라, 주어진 전이 확률이 유지된다는 **모델 안에서의 결과**입니다.")

    st.subheader("왜 A가 60%일까? : 흐름의 균형")
    c1, c2 = st.columns(2)
    c1.metric("A → G 흐름", "0.6 × 0.2 = 0.12")
    c2.metric("G → A 흐름", "0.4 × 0.3 = 0.12")
    st.success("두 흐름이 같으므로 빈도가 유지됩니다.")

# ---------- 5 ----------
elif page.startswith("5"):
    st.header("5. 탐구 과정 ③: 초기 빈도가 달라도 같은 값에 수렴할까?")
    st.write("전이 행렬은 $M_1$로 고정하고, 초기 A 빈도만 바꿔 비교합니다.")
    n = st.slider("세대 수", 5, 30, 15)
    d1 = simulate(0.2, 0.3, 0.9, n)
    d2 = simulate(0.2, 0.3, 0.2, n)
    st.plotly_chart(line_chart([d1, d2], ["초기 A 90%", "초기 A 20%"], 0.6),
                    use_container_width=True)

    t = pd.DataFrame({"세대": d2["세대"][:6],
                      "A 빈도(%)": (d2["A 빈도"][:6] * 100).round(3),
                      "G 빈도(%)": (d2["G 빈도"][:6] * 100).round(3)})
    st.write("초기 A 20%, G 80%일 때 (처음 5세대)")
    st.dataframe(t, hide_index=True)

    st.success("두 경우 모두 A 60%로 수렴합니다. 이 행렬에서는 초기 빈도와 무관하게 같은 정상 분포에 수렴합니다.")
    st.subheader("왜 그럴까?")
    st.latex(r"a_{n+1}=0.5a_n+0.3")
    st.latex(r"a_{n+1}-0.6=0.5\,(a_n-0.6)")
    st.write("정상값과의 차이가 매 세대 **절반**으로 줄어듭니다. (40%p → 20%p → 10%p → …)")

# ---------- 6 ----------
elif page.startswith("6"):
    st.header("6. 탐구 과정 ④: 전이 확률을 바꾸면?")
    c1, c2 = st.columns(2)
    c1.latex(r"M_1=\begin{pmatrix}0.8&0.2\\0.3&0.7\end{pmatrix}")
    c2.latex(r"M_2=\begin{pmatrix}0.9&0.1\\0.3&0.7\end{pmatrix}")
    st.latex(r"a=0.9a+0.3(1-a)\;\Rightarrow\;0.4a=0.3\;\Rightarrow\;\boxed{a=0.75}")

    st.table(pd.DataFrame({
        "원래 행렬 M₁": ["20%", "30%", "60%", "40%"],
        "변경 행렬 M₂": ["10%", "30%", "75%", "25%"],
    }, index=["A→G 확률", "G→A 확률", "A의 정상 빈도", "G의 정상 빈도"]))

    n = st.slider("세대 수", 5, 30, 15)
    d1 = simulate(0.2, 0.3, 0.9, n)
    d2 = simulate(0.1, 0.3, 0.9, n)
    fig = line_chart([d1, d2], ["M₁ (A→G 20%)", "M₂ (A→G 10%)"])
    fig.add_hline(y=60, line_dash="dot", line_color="gray")
    fig.add_hline(y=75, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    st.success("""
**발견 2**: 초기 빈도는 변화의 출발점에, 전이 확률은 정상 분포의 값에 영향을 줍니다.
다만 모든 마르코프 체인이 반드시 수렴하거나 정상 분포가 하나만 존재하는 것은 아닙니다.
""")

# ---------- 7 ----------
elif page.startswith("7"):
    st.header("7. 탐구 결과의 해석")
    st.markdown("""
**① 초기 빈도와 장기적인 빈도는 다를 수 있다.**
초기 빈도는 변화의 출발점이고, 장기 결과는 전이 확률의 영향을 받는다.

**② 전이 확률이 바뀌면 정상 분포도 바뀔 수 있다.**
A→G와 G→A 확률의 관계가 장기 빈도를 결정한다.
""")
    st.latex(r"a^*=\frac{p_{GA}}{p_{AG}+p_{GA}}")
    st.markdown("""
**③ 모델의 예측은 설정한 조건에 의존한다.**
실제 유전자 빈도는 자연선택, 집단 크기, 유전자 이동 등의 영향을 받으므로
전이 행렬 하나로 모든 변화를 설명할 수는 없다.
""")

# ---------- 8 ----------
elif page.startswith("8"):
    st.header("8. 생명과학 및 약학과의 연결")
    st.write("""
이 모델은 DNA 염기의 변화를 **확률적으로 분석**할 수 있음을 보여 줍니다.
생물정보학에서는 염기나 아미노산의 치환 확률로 유전자 서열의 진화적 관계를 분석합니다.
약학에서도 병원체의 유전적 변화를 이해하는 것은 약물 표적과 내성 연구에 중요할 수 있습니다.
""")
    st.warning("""
**한계**: A·G 두 상태 모델은 단순한 예시이므로 바이러스 변이나 약물 내성을 직접 예측할 수 없습니다.
실제 분석에는 내성 관련 변이, 변이의 생물학적 기능, 약물의 선택압 등의 정보가 추가로 필요합니다.
""")
    st.info("따라서 이 탐구의 의의는 확률 모델로 유전적 변화의 규칙을 분석하고, 이런 접근이 생물정보학에 활용됨을 이해한 데 있습니다.")

# ---------- 시뮬레이터 ----------
else:
    st.header("🧪 직접 해보는 시뮬레이터")
    c1, c2, c3 = st.columns(3)
    p_ag = c1.slider("A → G 확률", 0.0, 1.0, 0.2, 0.01)
    p_ga = c2.slider("G → A 확률", 0.0, 1.0, 0.3, 0.01)
    a0 = c3.slider("초기 A 빈도", 0.0, 1.0, 0.9, 0.01)
    n = st.slider("세대 수", 1, 50, 20)

    st.latex(rf"M=\begin{{pmatrix}}{1-p_ag:.2f}&{p_ag:.2f}\\{p_ga:.2f}&{1-p_ga:.2f}\end{{pmatrix}}")

    sd = stationary(p_ag, p_ga)
    df = simulate(p_ag, p_ga, a0, n)
    if sd is None:
        st.error("두 확률이 모두 0이면 어떤 염기도 바뀌지 않아 정상 분포가 유일하지 않습니다. 초기 빈도가 그대로 유지됩니다.")
        st.plotly_chart(line_chart([df], ["A 빈도"]), use_container_width=True)
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("정상 분포 A", f"{sd[0]*100:.2f}%")
        m2.metric("정상 분포 G", f"{sd[1]*100:.2f}%")
        lam = 1 - p_ag - p_ga
        m3.metric("수렴 속도 계수 (1−p−q)", f"{lam:.2f}")
        st.plotly_chart(line_chart([df], ["A 빈도"], sd[0]), use_container_width=True)
        if abs(lam) == 1:
            st.warning("계수의 절댓값이 1이라 수렴하지 않습니다. (A→G와 G→A가 모두 1이면 매 세대 A와 G가 서로 바뀝니다.)")
        elif lam < 0:
            st.info("계수가 음수이면 빈도가 정상값 주위를 오가며 진동하면서 수렴합니다.")
        st.dataframe(df.round(4), hide_index=True, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("※ 전이 확률은 가상의 값입니다.")
