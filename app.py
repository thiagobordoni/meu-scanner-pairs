import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from itertools import combinations
import plotly.graph_objects as go

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Trader Estatístico", layout="wide")

# 2. CSS ESTILO APPLE PREMIUM
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
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
    div.stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 3. CABEÇALHO CENTRALIZADO
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        # Puxa o arquivo logo.jpg que já está no seu GitHub
        st.image("logo.jpg", use_container_width=True)
    except:
        pass
    
    st.markdown("<h1 style='text-align: center; font-size: 2.5rem;'>Pair Trading Scanner</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #86868b; font-size: 1.2rem;'>Trader Estatístico • Cointegração Pro</p>", unsafe_allow_html=True)

st.markdown("---")

# 4. UNIVERSO DE ATIVOS
market_universe = {
    'Tech': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'META', 'GOOGL'],
    'Finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS'],
    'Energy': ['XOM', 'CVX', 'OXY', 'SLB']
}

# 5. SIDEBAR E PARÂMETROS
st.sidebar.header("📊 Parâmetros")
bp_limit = st.sidebar.number_input("Buying Power ($)", value=1000)
z_threshold = st.sidebar.slider("Z-Score Entrada", 1.0, 3.0, 1.96)

# 6. LÓGICA DO SCANNER
if st.sidebar.button("RODAR SCANNER AGORA"):
    all_tickers = [t for sub in market_universe.values() for t in sub]
    with st.spinner('Analisando o mercado...'):
        data = yf.download(all_tickers, period='350d', progress=False)['Close'].dropna(axis=1)

    results = []
    for sector, tickers in market_universe.items():
        available = [t for t in tickers if t in data.columns]
        for s1, s2 in combinations(available, 2):
            try:
                p_val = coint(data[s1].tail(250), data[s2].tail(250))[1]
                if p_val < 0.05:
                    y, x = data[s1].tail(250).values, sm.add_constant(data[s2].tail(250).values)
                    model = sm.OLS(y, x).fit()
                    beta = model.params[1]
                    res = data[s1].tail(250) - (model.params[0] + beta * data[s2].tail(250))
                    z = (res.iloc[-1] - res.mean()) / res.std()
                    
                    if abs(z) >= z_threshold:
                        results.append({"Par": f"{s1}/{s2}", "Z-Score": round(z, 2), "Beta": round(beta, 2)})
            except: continue

    if results:
        st.success("Oportunidades encontradas!")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.warning("Nenhuma distorção encontrada.")
