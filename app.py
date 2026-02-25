import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from itertools import combinations
from datetime import datetime
import plotly.graph_objects as go

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Trader Estatístico", layout="wide", initial_sidebar_state="expanded")

# --- CSS ESTILO APPLE PREMIUM ---
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #f5f5f7; font-family: -apple-system, sans-serif; }
    [data-testid="stSidebar"] { background-color: #1c1c1e; border-right: 1px solid #333; }
    .stButton>button { border-radius: 20px; background-color: #007AFF; color: white; border: none; font-weight: 500; transition: 0.3s; }
    .stButton>button:hover { background-color: #0056b3; transform: scale(1.02); }
    div.stDataFrame { background: rgba(255, 255, 255, 0.05); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); }
    h1 { font-weight: 600 !important; letter-spacing: -0.5px !important; }
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- LOGIN ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.markdown("<div style='padding: 20px; background: #1c1c1e; border-radius: 20px; margin-top: 50px;'>", unsafe_allow_html=True)
            pwd = st.text_input("Acesso Restrito", type="password", placeholder="Senha")
            if st.button("Entrar"):
                if pwd == "SUA_SENHA_AQUI": # Defina sua senha aqui
                    st.session_state["password_correct"] = True
                    st.rerun()
                else: st.error("Incorreto.")
            st.markdown("</div>", unsafe_allow_html=True)
        return False
    return True

if not check_password(): st.stop()

# --- CABEÇALHO ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try: st.image("logo.jpg", use_container_width=True)
    except: st.info("Suba o logo.jpg no GitHub.")
    st.markdown("<h1 style='text-align: center;'>Pair Trading Scanner</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #86868b;'>Trader Estatístico • Cointegração em Tempo Real</p>", unsafe_allow_html=True)

st.markdown("---")

# 1. UNIVERSO DE ATIVOS
market_universe = {
    'Tech': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'META', 'GOOGL', 'PLTR', 'UBER'],
    'Finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'PYPL'],
    'Consumer': ['KO', 'PEP', 'SBUX', 'NKE', 'TGT', 'WMT', 'COST'],
    'Energy': ['XOM', 'CVX', 'OXY', 'SLB', 'HAL']
}

# 2. FUNÇÕES TÉCNICAS
def get_zscore_series(s1_data, s2_data, window=250):
    y, x = s1_data.tail(window).values, sm.add_constant(s2_data.tail(window).values)
    model = sm.OLS(y, x).fit()
    beta = model.params[1]
    residue = s1_data.tail(window) - (model.params[0] + beta * s2_data.tail(window))
    return (residue - residue.mean()) / residue.std(), beta

# 3. SIDEBAR
st.sidebar.header("📊 Configurações")
z_thresh = st.sidebar.slider("Z-Score Entrada", 1.0, 3.0, 1.96)
risk_usd = st.sidebar.number_input("Risco Alvo ($)", value=15)

if st.sidebar.button("RODAR SCANNER"):
    all_tickers = [t for sub in market_universe.values() for t in sub]
    with st.spinner('Escaneando distorções estatísticas...'):
        data = yf.download(all_tickers, period='350d', progress=False)['Close'].dropna(axis=1)

    results = []
    for sector, tickers in market_universe.items():
        available = [t for t in tickers if t in data.columns]
        for s1, s2 in combinations(available, 2):
            try:
                # Teste simplificado de cointegração para performance
                p_val = coint(data[s1].tail(250), data[s2].tail(250))[1]
                if p_val < 0.05:
                    z_series, beta = get_zscore_series(data[s1], data[s2])
                    z_last = z_series.iloc[-1]
                    
                    if abs(z_last) >= z_thresh:
                        results.append({
                            "Par": f"{s1}/{s2}",
                            "Sinal": "VENDE/COMPRA" if z_last > 0 else "COMPRA/VENDE",
                            "Z": round(z_last, 2),
                            "Beta": round(beta, 2),
                            "S1": s1, "S2": s2
                        })
            except: continue

    if results:
        df = pd.DataFrame(results)
        st.success(f"{len(df)} Oportunidades!")
        
        # Tabela interativa
        selected_row = st.selectbox("Selecione um par para visualizar o gráfico:", df["Par"])
        st.dataframe(df.drop(columns=["S1", "S2"]), use_container_width=True)
        
        # Gráfico Dinâmico Plotly
        row = df[df["Par"] == selected_row].iloc[0]
        z_plot, _ = get_zscore_series(data[row["S1"]], data[row["S2"]])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=z_plot, mode='lines', name='Z-Score', line=dict(color='#007AFF', width=2.5)))
        fig.add_hline(y=z_thresh, line_dash="dash", line_color="#FF3B30", annotation_text="Entrada")
        fig.add_hline(y=-z_thresh, line_dash="dash", line_color="#FF3B30")
        fig.add_hline(y=0, line_color="white", opacity=0.3)
        
        fig.update_layout(
            title=f"Histórico de Desvio: {selected_row}",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f5f5f7', height=400,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Nenhuma oportunidade encontrada.")
