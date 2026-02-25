import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from itertools import combinations
from datetime import datetime
import plotly.graph_objects as go

# 1. CONFIGURAÇÃO ÚNICA DA PÁGINA
st.set_page_config(page_title="Trader Estatístico", layout="wide", initial_sidebar_state="expanded")

# 2. CSS ESTILO APPLE PREMIUM (DARK MODE)
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #1c1c1e;
        border-right: 1px solid #333;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #007AFF;
        color: white;
        border: none;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: scale(1.02);
    }
    div.stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 3. SISTEMA DE LOGIN PRIVADO
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.markdown("<div style='padding: 20px; background: #1c1c1e; border-radius: 20px; margin-top: 50px;'>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>Acesso Restrito</h2>", unsafe_allow_html=True)
            pwd = st.text_input("Senha", type="password", placeholder="Digite a senha mestra")
            if st.button("Entrar"):
                if pwd == "1234": # DEFINA SUA SENHA AQUI
                    st.session_state["password_correct"] = True
                    st.rerun()
                else:
                    st.error("Senha incorreta.")
            st.markdown("</div>", unsafe_allow_html=True)
        return False
    return True

if not check_password():
    st.stop()

# 4. CABEÇALHO CENTRALIZADO COM A FOTO LOGO.JPG
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        # Puxa o arquivo logo.jpg que você subiu (1.57 MB)
        st.image("logo.jpg", use_container_width=True)
    except:
        st.info("⚠️ Verifique se o arquivo se chama exatamente logo.jpg no GitHub.")
    
    st.markdown("<h1 style='text-align: center; font-size: 2.5rem;'>Pair Trading Scanner</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #86868b; font-size: 1.2rem;'>Trader Estatístico • Cointegração em Tempo Real</p>", unsafe_allow_html=True)

st.markdown("---")

# 5. UNIVERSO DE ATIVOS
market_universe = {
    'Tech_Mid_Small': ['AAL', 'CCL', 'F', 'PLTR', 'SNAP', 'PINS', 'UBER', 'HOOD', 'AFRM', 'SOFI', 'DKNG', 'U', 'RBLX', 'RIOT', 'MARA', 'DBX', 'BOX', 'GPRO', 'CLOV', 'MQ', 'COIN'],
    'Financials': ['BAC', 'WFC', 'C', 'AIG', 'PFG', 'TFC', 'KEY', 'HBAN', 'RF', 'FITB', 'SLM', 'NYCB', 'SQ', 'PYPL', 'NU', 'PGR', 'MET', 'USB', 'SCHW'],
    'Consumer_Retail': ['KO', 'PEP', 'TGT', 'KSS', 'M', 'AEO', 'SBUX', 'NCLH', 'RCL', 'GPS', 'JWN', 'WEN', 'QSR', 'CPRI', 'DLTR', 'LUV'],
    'Energy_Materials': ['XOM', 'CVX', 'OXY', 'HAL', 'SLB', 'KOS', 'RIG', 'APA', 'DVN', 'CLF', 'FCX', 'NEM', 'VALE', 'X', 'AA', 'MOS', 'CTRA']
}

# 6. FUNÇÕES TÉCNICAS
def calculate_half_life(residue):
    delta_residue = residue.diff().dropna()
    lagged_residue = residue.shift(1).dropna()
    if len(lagged_residue) < 10: return 999
    x = sm.add_constant(lagged_residue.values)
    model = sm.OLS(delta_residue.values, x).fit()
    lambda_val = model.params[1]
    return -np.log(2) / lambda_val if lambda_val < 0 else 999

# 7. INTERFACE LATERAL
st.sidebar.header("📊 Parâmetros")
bp_limit = st.sidebar.number_input("Buying Power por Par ($)", value=1000)
z_threshold = st.sidebar.slider("Z-Score de Entrada", 1.0, 3.0, 1.96)
risk_target = st.sidebar.number_input("Risco Alvo ($)", value=15)

# 8. EXECUÇÃO DO SCANNER
if st.sidebar.button("RODAR SCANNER AGORA"):
    all_tickers = [t for sub in market_universe.values() for t in sub]
    with st.spinner('Analisando mercado estatístico...'):
        data = yf.download(all_tickers, period='350d', progress=False)['Close'].dropna(axis=1)

    results = []
    progress_bar = st.progress(0)
    sectors = list(market_universe.keys())

    for i, sector in enumerate(sectors):
        available = [t for t in market_universe[sector] if t in data.columns]
        for s1, s2 in combinations(available, 2):
            try:
                # Teste de Cointegração (Simplified)
                p_val = coint(data[s1].tail(250), data[s2].tail(250))[1]
                if p_val < 0.05:
                    y, x = data[s1].tail(250).values, sm.add_constant(data[s2].tail(250).values)
                    model = sm.OLS(y, x).fit()
                    beta = model.params[1]
                    if beta < 0.20: continue
                    
                    residue = data[s1].tail(250) - (model.params[0] + beta * data[s2].tail(250))
                    z_score = (residue.iloc[-1] - residue.mean()) / residue.std()
                    
                    if abs(z_score) >= z_threshold:
                        results.append({
                            "Setor": sector, "Par": f"{s1}/{s2}", 
                            "Beta": round(beta, 2), "Z": round(z_score, 2),
                            "Sinal": "VENDE/COMPRA" if z_score > 0 else "COMPRA/VENDE"
                        })
            except: continue
        progress_bar.progress((i + 1) / len(sectors))

    if results:
        df_final = pd.DataFrame(results).sort_values(by='Z', ascending=False, key=abs)
        st.success(f"Oportunidades encontradas!")
        st.dataframe(df_final, use_container_width=True)
    else:
        st.warning("Nenhuma oportunidade encontrada.")

st.sidebar.markdown("---")
st.sidebar.info("Saia da operação quando o Z chegar em 0.00.")
