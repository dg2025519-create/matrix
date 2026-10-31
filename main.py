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
    "8. 생명과학 연결",
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
    st.write("이 탐구에 쓰이는 수학 개념을 하나씩 정리합니다. 순서대로 읽으면 3단계 계산이 자연스럽게 이해됩니다.")

    tabs = st.tabs(["① 마르코프 체인", "② 행렬과 벡터", "③ 행렬의 곱셈",
                    "④ 전치행렬", "⑤ 전이행렬", "⑥ 빈도 벡터와 세대 계산", "⑦ 정상 분포"])

    with tabs[0]:
        st.subheader("마르코프 체인 (Markov chain)")
        st.write("""
**상태(state)** 는 시스템이 가질 수 있는 상황입니다. 이 탐구에서는 한 위치의 염기가
**A인 상태**, **G인 상태** 두 가지입니다. (계산을 간단히 하려고 A·G만 고려합니다.)

마르코프 체인은 **다음 상태가 오직 현재 상태에만 의존**한다고 가정하는 확률 모델입니다.
과거에 어떤 염기였는지는 상관없고, 지금 무엇인지만 알면 다음 세대의 확률을 계산할 수 있습니다.
이 성질을 **마르코프 성질**이라고 합니다.
""")
        st.latex(r"P(X_{n+1}\mid X_n, X_{n-1}, \dots, X_0)=P(X_{n+1}\mid X_n)")
        st.caption("읽는 법: 다음 상태의 확률은 지금까지의 모든 과거를 알 때나, 현재만 알 때나 같다.")
        st.info("예: 오늘 날씨만 알면 내일 비 올 확률을 구할 수 있다고 보는 것과 같은 아이디어입니다.")

    with tabs[1]:
        st.subheader("행렬(matrix)과 벡터(vector)")
        st.write("""
**행렬**은 수를 직사각형 모양으로 배열한 것입니다. 가로줄을 **행(row)**, 세로줄을 **열(column)** 이라 합니다.
행이 $m$개, 열이 $n$개이면 $m\\times n$ 행렬이라 합니다.
""")
        st.latex(r"A=\begin{pmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{pmatrix}\quad (2\times2\ \text{행렬})")
        st.write("$a_{ij}$는 **i행 j열** 의 성분입니다. 예를 들어 $a_{12}$는 1행 2열의 수입니다.")

        st.write("**열벡터**는 열이 1개인 행렬(세로로 쓴 수의 나열), **행벡터**는 행이 1개인 행렬(가로로 쓴 수의 나열)입니다.")
        c1, c2 = st.columns(2)
        c1.latex(r"\text{열벡터: }\ v=\begin{pmatrix}0.9\\0.1\end{pmatrix}\ (2\times1)")
        c2.latex(r"\text{행벡터: }\ v^{T}=\begin{pmatrix}0.9&0.1\end{pmatrix}\ (1\times2)")
        st.info("이 탐구에서는 염기 빈도를 **열벡터**로 씁니다. 위쪽 성분이 A의 빈도, 아래쪽 성분이 G의 빈도입니다.")

    with tabs[2]:
        st.subheader("행렬의 곱셈")
        st.write("""
행렬 곱은 **(왼쪽 행렬의 행) × (오른쪽 행렬의 열)** 을 성분끼리 곱해 더하는 방식입니다.
2×2 행렬과 2×1 열벡터의 곱은 다음과 같습니다.
""")
        st.latex(r"""
\begin{pmatrix}a&b\\c&d\end{pmatrix}
\begin{pmatrix}x\\y\end{pmatrix}
=\begin{pmatrix}ax+by\\cx+dy\end{pmatrix}
""")
        st.write("**수치 예시**")
        st.latex(r"""
\begin{pmatrix}0.8&0.3\\0.2&0.7\end{pmatrix}
\begin{pmatrix}0.9\\0.1\end{pmatrix}
=\begin{pmatrix}0.8\times0.9+0.3\times0.1\\0.2\times0.9+0.7\times0.1\end{pmatrix}
=\begin{pmatrix}0.75\\0.25\end{pmatrix}
""")
        st.warning("곱셈이 가능하려면 **왼쪽 행렬의 열 수 = 오른쪽 행렬의 행 수** 여야 합니다. 또한 행렬 곱은 순서를 바꾸면 결과가 달라질 수 있습니다.")

    with tabs[3]:
        st.subheader("전치행렬 (transpose)")
        st.write("""
전치행렬은 **행과 열을 서로 바꾼 행렬**입니다. 기호는 $M^T$로 씁니다.
$i$행 $j$열의 성분이 $j$행 $i$열로 이동합니다. 대각선(왼쪽 위 → 오른쪽 아래)을 기준으로 뒤집는 것과 같습니다.
""")
        st.latex(r"""
M=\begin{pmatrix}0.8&0.2\\0.3&0.7\end{pmatrix}
\quad\Longrightarrow\quad
M^{T}=\begin{pmatrix}0.8&0.3\\0.2&0.7\end{pmatrix}
""")
        st.write("대각선 위의 성분(0.8, 0.7)은 그대로이고, 나머지 두 성분(0.2와 0.3)의 위치가 서로 바뀝니다.")

        st.markdown("**왜 이 탐구에서 전치행렬이 필요할까?**")
        st.write("""
전이행렬 $M$은 **행 = 현재 상태, 열 = 다음 상태** 로 정의했습니다. 예를 들어 $M$의 1행 2열(0.2)은
"현재 A → 다음 G"의 확률입니다. 그런데 다음 세대의 A 빈도를 구하려면
**"다음 상태가 A"인 확률들(= M의 1열)** 을 현재 빈도와 곱해 더해야 합니다.

$M$의 **열**을 **행**으로 바꿔 주면 위에서 배운 "행 × 열벡터" 곱셈을 그대로 쓸 수 있습니다. 그래서 $M^T$를 곱합니다.
""")
        st.latex(r"""
\underbrace{\begin{pmatrix}0.8&0.3\\0.2&0.7\end{pmatrix}}_{M^{T}}
\underbrace{\begin{pmatrix}A\text{의 빈도}\\G\text{의 빈도}\end{pmatrix}}_{v_n}
=\begin{pmatrix}0.8\,A+0.3\,G\\0.2\,A+0.7\,G\end{pmatrix}
""")
        st.caption("첫 줄 = (A→A 확률)×A빈도 + (G→A 확률)×G빈도 = 다음 세대 A의 빈도")

        st.markdown("**직접 확인해 보기**")
        cc = st.columns(4)
        m11 = cc[0].number_input("M의 1행1열", 0.0, 1.0, 0.8, 0.1, key="t11")
        m12 = cc[1].number_input("M의 1행2열", 0.0, 1.0, 0.2, 0.1, key="t12")
        m21 = cc[2].number_input("M의 2행1열", 0.0, 1.0, 0.3, 0.1, key="t21")
        m22 = cc[3].number_input("M의 2행2열", 0.0, 1.0, 0.7, 0.1, key="t22")
        st.latex(rf"M=\begin{{pmatrix}}{m11}&{m12}\\{m21}&{m22}\end{{pmatrix}}"
                 rf"\ \Rightarrow\ M^T=\begin{{pmatrix}}{m11}&{m21}\\{m12}&{m22}\end{{pmatrix}}")

    with tabs[4]:
        st.subheader("전이확률행렬 (transition matrix)")
        st.write("""
상태 사이를 이동할 **확률**을 모아 놓은 행렬입니다. $i$행 $j$열의 성분 $p_{ij}$는
**"현재 상태 $i$에서 다음 상태 $j$로 바뀔 확률"** 입니다.
""")
        st.table(pd.DataFrame({"다음 A": ["p_AA = 0.8", "p_GA = 0.3"],
                               "다음 G": ["p_AG = 0.2", "p_GG = 0.7"]},
                              index=["현재 A", "현재 G"]))
        st.latex(r"M=\begin{pmatrix}p_{AA}&p_{AG}\\p_{GA}&p_{GG}\end{pmatrix}=\begin{pmatrix}0.8&0.2\\0.3&0.7\end{pmatrix}")

        st.markdown("**전이행렬의 두 가지 조건**")
        st.markdown("""
1. 모든 성분은 확률이므로 **0 이상 1 이하**
2. **각 행의 합은 1** : 현재 A인 염기는 다음 세대에 반드시 A 또는 G가 되므로
   $p_{AA}+p_{AG}=1$, $p_{GA}+p_{GG}=1$
""")
        st.warning("행의 합이 1이지, 열의 합이 1인 것은 아닙니다. 위 행렬의 1열 합은 0.8+0.3=1.1입니다.")
        st.write("이런 조건을 만족하는 행렬을 **확률행렬(stochastic matrix)** 또는 **행 확률행렬**이라고 합니다.")

    with tabs[5]:
        st.subheader("염기 빈도 벡터와 세대 계산")
        st.write("""
집단에서 A와 G가 차지하는 비율을 열벡터로 나타낸 것이 **염기 빈도 벡터**입니다.
비율이므로 성분의 합은 항상 1입니다.
""")
        st.latex(r"v_n=\begin{pmatrix}a_n\\g_n\end{pmatrix},\qquad a_n+g_n=1")
        st.write("한 세대가 지날 때마다 전치행렬을 곱합니다.")
        st.latex(r"v_{n+1}=M^{T}v_n")
        st.write("이를 반복하면 $n$세대 뒤의 빈도는 다음과 같이 쓸 수 있습니다.")
        st.latex(r"v_n=(M^{T})^{n}\,v_0")
        st.caption("$(M^T)^n$은 $M^T$를 n번 곱한 행렬입니다.")

        st.markdown("**직접 계산해 보기**")
        a_in = st.slider("현재 A 빈도", 0.0, 1.0, 0.9, 0.05, key="vin")
        g_in = 1 - a_in
        a_out = 0.8 * a_in + 0.3 * g_in
        g_out = 0.2 * a_in + 0.7 * g_in
        st.latex(rf"""
\begin{{pmatrix}}0.8&0.3\\0.2&0.7\end{{pmatrix}}
\begin{{pmatrix}}{a_in:.2f}\\{g_in:.2f}\end{{pmatrix}}
=\begin{{pmatrix}}{a_out:.3f}\\{g_out:.3f}\end{{pmatrix}}
""")
        st.caption("빈도의 합이 계속 1로 유지되는 것을 확인해 보세요. (행의 합이 1인 전이행렬을 썼기 때문입니다.)")

    with tabs[6]:
        st.subheader("정상 분포 (stationary distribution)")
        st.write("전이행렬을 한 번 더 적용해도 **변하지 않는 빈도 벡터** $\\pi$를 정상 분포라 합니다.")
        st.latex(r"M^{T}\pi=\pi")
        st.write("""
선형대수 용어로는 $M^T$의 **고유값 1에 대응하는 고유벡터**(성분의 합을 1로 맞춘 것)입니다.
고등학교 수준에서는 "한 번 곱해도 그대로인 벡터"로 이해하면 충분합니다.
""")
        st.markdown("**2×2 전이행렬의 정상 분포 공식**")
        st.latex(r"M=\begin{pmatrix}1-p&p\\q&1-q\end{pmatrix}\ \Rightarrow\ \pi=\begin{pmatrix}\dfrac{q}{p+q}\\[2mm]\dfrac{p}{p+q}\end{pmatrix}")
        st.caption("p = A→G 확률, q = G→A 확률. p=0.2, q=0.3이면 A는 0.3/0.5 = 0.6")

        st.markdown("**직접 확인: $M^T\\pi=\\pi$ 인가?**")
        pi = np.array([0.6, 0.4])
        Mt = np.array([[0.8, 0.3], [0.2, 0.7]])
        st.latex(rf"""
\begin{{pmatrix}}0.8&0.3\\0.2&0.7\end{{pmatrix}}
\begin{{pmatrix}}0.6\\0.4\end{{pmatrix}}
=\begin{{pmatrix}}{(Mt@pi)[0]:.2f}\\{(Mt@pi)[1]:.2f}\end{{pmatrix}}
=\begin{{pmatrix}}0.6\\0.4\end{{pmatrix}}
""")
        st.success("계산 결과가 원래 벡터와 같으므로 (0.6, 0.4)는 정상 분포입니다.")

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
    st.header("8. 생명과학 연결")
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
    st.write("슬라이더를 움직여 **수렴하지 않는 경우**를 찾아보세요. "
             "힌트: 수렴 여부는 λ = 1 − p − q 의 절댓값이 결정합니다.")

    preset = st.radio("빠른 설정", ["직접 조절", "탐구 기본값 (0.2, 0.3)", "진동하며 수렴 (0.7, 0.8)",
                                  "수렴 안 함: 완전 교환 (1, 1)", "수렴 안 함: 변화 없음 (0, 0)"],
                      horizontal=True)
    presets = {"탐구 기본값 (0.2, 0.3)": (0.2, 0.3), "진동하며 수렴 (0.7, 0.8)": (0.7, 0.8),
               "수렴 안 함: 완전 교환 (1, 1)": (1.0, 1.0), "수렴 안 함: 변화 없음 (0, 0)": (0.0, 0.0)}

    c1, c2, c3 = st.columns(3)
    if preset == "직접 조절":
        p_ag = c1.slider("A → G 확률", 0.0, 1.0, 0.2, 0.01)
        p_ga = c2.slider("G → A 확률", 0.0, 1.0, 0.3, 0.01)
    else:
        p_ag, p_ga = presets[preset]
        c1.metric("A → G 확률", p_ag)
        c2.metric("G → A 확률", p_ga)
    a0 = c3.slider("초기 A 빈도", 0.0, 1.0, 0.9, 0.01)
    n = st.slider("세대 수", 1, 50, 20)

    st.latex(rf"M=\begin{{pmatrix}}{1-p_ag:.2f}&{p_ag:.2f}\\{p_ga:.2f}&{1-p_ga:.2f}\end{{pmatrix}}")

    lam = 1 - p_ag - p_ga
    sd = stationary(p_ag, p_ga)
    df = simulate(p_ag, p_ga, a0, n)

    if sd is None:
        st.error("p = q = 0 (λ = 1): 어떤 염기도 바뀌지 않아 빈도가 초기값 그대로입니다. "
                 "정상 분포가 유일하지 않고, 초기값에 따라 결과가 달라집니다.")
        st.plotly_chart(line_chart([df], ["A 빈도"]), use_container_width=True)
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("정상 분포 A", f"{sd[0]*100:.2f}%")
        m2.metric("정상 분포 G", f"{sd[1]*100:.2f}%")
        m3.metric("λ = 1 − p − q", f"{lam:.2f}")
        st.plotly_chart(line_chart([df], ["A 빈도"], sd[0]), use_container_width=True)

        if abs(lam + 1) < 1e-9:
            if abs(a0 - 0.5) < 1e-9:
                st.warning("p = q = 1 (λ = −1): 초기 A가 50%라서 우연히 계속 50%로 유지됩니다. "
                           "초기값을 바꾸면 진동하는 것을 볼 수 있습니다.")
            else:
                st.error("p = q = 1 (λ = −1): 매 세대 A와 G가 완전히 뒤바뀌어 "
                         "빈도가 두 값을 오가기만 하고 **수렴하지 않습니다.**")
        elif lam < 0:
            st.info("λ가 음수: 정상값 주위를 오가며 진동하지만 점점 줄어들어 수렴합니다.")
        else:
            st.success("0 ≤ λ < 1: 정상값으로 한쪽 방향으로 수렴합니다.")
        st.dataframe(df.round(4), hide_index=True, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("※ 전이 확률은 가상의 값입니다.")
