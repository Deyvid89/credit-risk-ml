import os
import sys
import joblib
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.features import feature_engineering
from src.scoring import validate_inputs, build_features, calculate_score, score_label
from src.rules import apply_business_rules

# ==============================
# CONFIGURAÇÃO
# ==============================
st.set_page_config(
    page_title="CreditIQ · Análise de Risco",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================
# ESTILO
# ==============================
def load_css(path: str):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(os.path.join(os.path.dirname(__file__), "style.css"))

# ==============================
# MODELO
# ==============================
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# ==============================
# HELPERS DE VISUALIZAÇÃO
# ==============================
def plotly_bar(value: float, label: str):
    """Barra de progresso horizontal estilizada via Plotly."""
    if value > 0.8:
        bar_color, text_color = "#f04b4b", "#1a0000"
    elif value > 0.5:
        bar_color, text_color = "#f59e0b", "#1a1200"
    else:
        bar_color, text_color = "#22c27a", "#003320"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[1], y=[label], orientation="h",
        marker_color="#1c2030", width=0.7, showlegend=False,
    ))
    fig.add_trace(go.Bar(
        x=[value], y=[label], orientation="h",
        marker_color=bar_color, width=0.7, showlegend=False,
        text=f"{value:.0%}", textposition="inside",
        textfont=dict(color=text_color, size=12, family="DM Sans", weight="bold"),
    ))
    fig.update_layout(
        barmode="overlay", height=38,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 1], visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def build_gauge(score: int) -> go.Figure:
    """Gauge de score de crédito."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Score", "font": {"color": "#e8eaf0", "family": "DM Sans"}},
        number={"font": {"color": "#e8eaf0", "family": "DM Serif Display", "size": 36}},
        gauge={
            "axis": {"range": [0, 1000], "tickcolor": "#7a8099",
                     "tickfont": {"color": "#7a8099", "size": 10}},
            "steps": [
                {"range": [0,   430],  "color": "#ff4d4f"},
                {"range": [430, 600],  "color": "#faad14"},
                {"range": [600, 760],  "color": "#1890ff"},
                {"range": [760, 1000], "color": "#52c41a"},
            ],
            "bar": {"color": "white", "thickness": 0.25},
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        }
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=30, r=30, t=50, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eaf0",
    )
    return fig


def render_alerts(triggered: list):
    """Renderiza os alertas com scroll se houver mais de 4."""
    if triggered:
        alerts_html = ""
        for kind, msg in triggered:
            css = "alert-err" if kind == "err" else "alert-warn"
            alerts_html += f'<div class="alert-item {css}">{msg}</div>'
        if len(triggered) > 4:
            alerts_html = (
                f'<div style="max-height:200px;overflow-y:auto;padding-right:4px;">'
                f'{alerts_html}</div>'
            )
        st.markdown(alerts_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="alert-item alert-ok">✅ Nenhum alerta identificado</div>',
            unsafe_allow_html=True,
        )

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="app-header">
    <p class="logo">Credit<span>IQ</span></p>
    <p class="tagline">Análise de Risco de Crédito · Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# LAYOUT: formulário | resultado
# ==============================
col_form, col_result = st.columns([1, 1], gap="large")

# ── FORMULÁRIO ───────────────────────────────────────────
with col_form:
    with st.form("risk_form"):

        st.markdown('<div class="section-title">👤 Perfil</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age         = st.number_input("Idade", 18, 100, 45)
            open_credit = st.number_input("Linhas de crédito", 0, 50, 6)
        with c2:
            monthly_income = st.number_input("Renda mensal (R$)", 0.0, value=3500.0, step=100.0)
            real_estate    = st.number_input("Emprést. imobiliários", 0, 10, 1)
        dependents = st.number_input("Dependentes", 0, 10, 2)

        st.markdown('<div class="section-title">💰 Exposição</div>', unsafe_allow_html=True)
        debt_ratio         = st.slider("Taxa de endividamento", 0.0, 1.0, 0.60, format="%.2f")
        credit_utilization = st.slider("Uso de crédito",        0.0, 1.0, 0.75, format="%.2f")

        st.markdown('<div class="section-title">📅 Histórico de Atrasos</div>', unsafe_allow_html=True)
        ca, cb, cc = st.columns(3)
        with ca: late_30 = st.number_input("30–59 dias", 0, 50, 1)
        with cb: late_60 = st.number_input("60–89 dias", 0, 50, 0)
        with cc: late_90 = st.number_input("90+ dias",   0, 50, 1)

        submitted = st.form_submit_button("🔍 Analisar Risco de Crédito")

# ── RESULTADO ────────────────────────────────────────────
with col_result:
    if not submitted:
        st.markdown("""
        <div style="height:100%;display:flex;flex-direction:column;
                    justify-content:center;align-items:center;
                    color:#3a3f55;text-align:center;padding:4rem 1rem;">
            <div style="font-size:2.5rem;margin-bottom:0.7rem;">💳</div>
            <div style="font-size:0.88rem;line-height:1.6;">
                Preencha o formulário ao lado e clique em<br>
                <strong style="color:#4f7cff">Analisar Risco</strong> para ver o score.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        inputs = {
            "credit_utilization": credit_utilization,
            "age":                age,
            "late_30":            late_30,
            "debt_ratio":         debt_ratio,
            "monthly_income":     monthly_income,
            "open_credit_lines":  open_credit,
            "late_90":            late_90,
            "real_estate_loans":  real_estate,
            "late_60":            late_60,
            "dependents":         dependents,
        }

        errors = validate_inputs(inputs)
        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        features = build_features(inputs)

        with st.spinner("Calculando..."):
            try:
                prob_raw = float(model.predict_proba(features)[0][1])
            except Exception as e:
                st.error(f"Erro: {e}")
                st.stop()

        prob, triggered      = apply_business_rules(prob_raw, inputs)
        score                = calculate_score(prob)
        label, color, pill_class = score_label(score)

        # Score banner
        st.markdown(f"""
        <div class="score-banner">
            <div class="score-num">{score}</div>
            <div><span class="pill {pill_class}">{label}</span></div>
            <div class="score-prob">Probabilidade de inadimplência: <strong>{prob:.1%}</strong></div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge
        st.plotly_chart(build_gauge(score), use_container_width=True)

        # Sub-colunas: indicadores | alertas
        sub_ind, sub_alert = st.columns([1, 1], gap="medium")

        with sub_ind:
            st.markdown('<div class="section-title">📊 Indicadores</div>', unsafe_allow_html=True)
            st.caption("Endividamento")
            st.plotly_chart(plotly_bar(debt_ratio, "Endividamento"), use_container_width=True)
            st.caption("Uso de Crédito")
            st.plotly_chart(plotly_bar(credit_utilization, "Uso de Crédito"), use_container_width=True)

        with sub_alert:
            st.markdown('<div class="section-title">⚠️ Alertas</div>', unsafe_allow_html=True)
            render_alerts(triggered)

        # KPIs
        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        k1.metric("Renda Mensal",     f"R$ {monthly_income:,.0f}")
        k2.metric("Total de Atrasos", late_30 + late_60 + late_90)
        k3.metric("Dependentes",      dependents)
