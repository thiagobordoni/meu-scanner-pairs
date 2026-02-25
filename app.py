import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from itertools import combinations
from datetime import datetime

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Trader Estatístico", layout="wide", initial_sidebar_state="expanded")

# --- CSS CUSTOMIZADO (ESTILO APPLE) ---
st.markdown("""
    <style>
    /* Fundo principal e fontes */
    .stApp {
        background-color: #000000;
        color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar minimalista */
    [data-testid="stSidebar"] {
        background-color: #1c1c1e;
        border-right: 1px solid #333;
    }
    
    /* Botões estilo Apple */
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

    /* Cards e Containers */
    div.stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Títulos e Textos */
    h1 {
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }
    
    /* Esconder o menu padrão do Streamlit para parecer um app real */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- CABEÇALHO ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        st.image("logo.jpg", use_container_width=True)
    except:
        pass
    st.markdown("<h1 style='text-align: center; font-size: 2.5rem;'>Pair Trading Scanner</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #86868b; font-size: 1.2rem;'>Trader Estatístico • Cointegração em Tempo Real</p>", unsafe_allow_html=True)

st.markdown("---")

# --- SISTEMA DE LOGIN PRIVADO ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.markdown("<div style='padding: 20px; background: #1c1c1e; border-radius: 20px;'>", unsafe_allow_html=True)
            pwd = st.text_input("Acesso Restrito", type="password", placeholder="Digite sua senha")
            if st.button("Entrar"):
                if pwd == "SUA_SENHA_AQUI": # Defina sua senha aqui
                    st.session_state["password_correct"] = True
                    st.rerun()
                else:
                    st.error("Senha incorreta.")
            st.markdown("</div>", unsafe_allow_html=True)
        return False
    return True

if not check_password():
    st.stop()

# (Mantenha o restante do seu código original de cálculos e market_universe aqui...)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from itertools import combinations
from datetime import datetime

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Trader Estatístico", layout="wide")

# --- CABEÇALHO COM IMAGEM CENTRALIZADA E TÍTULOS ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Tenta carregar a imagem se ela existir no repo
    try:
        st.image("logo.jpg", use_container_width=True)
    except:
        st.info("Suba o arquivo logo.jpg para o GitHub para exibir a imagem aqui.")
    
    st.markdown("<h1 style='text-align: center;'>Pair Trading Scanner (Cointegração)</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Trader Estatístico</h3>", unsafe_allow_html=True)

# 1. UNIVERSO DE ATIVOS (APENAS STOCKS)
market_universe = {
    'Tech_Mid_Small': ['AAL', 'CCL', 'F', 'PLTR', 'SNAP', 'PINS', 'UBER', 'HOOD', 'AFRM', 'SOFI', 'DKNG', 'U', 'RBLX', 'RIOT', 'MARA', 'DBX', 'BOX', 'GPRO', 'CLOV', 'MQ', 'COIN'],
    'Financials': ['BAC', 'WFC', 'C', 'AIG', 'PFG', 'TFC', 'KEY', 'HBAN', 'RF', 'FITB', 'SLM', 'NYCB', 'SQ', 'PYPL', 'NU', 'PGR', 'MET', 'USB', 'SCHW'],
    'Health_Care': ['CVS', 'PFE', 'BMY', 'GILD', 'VTRS', 'HLT', 'MRNA', 'BILI', 'TEVA', 'WBA', 'PZZA', 'PARA', 'RMNI'],
    'Energy_Materials': ['XOM', 'CVX', 'OXY', 'HAL', 'SLB', 'KOS', 'RIG', 'APA', 'DVN', 'CLF', 'FCX', 'NEM', 'VALE', 'X', 'AA', 'MOS', 'CTRA'],
    'Consumer_Retail': ['KO', 'PEP', 'TGT', 'KSS', 'M', 'AEO', 'SBUX', 'NCLH', 'RCL', 'GPS', 'JWN', 'WEN', 'QSR', 'CPRI', 'DLTR', 'LUV'],
    'Industrials_Transp': ['GE', 'CSX', 'NSC', 'DAL', 'UAL', 'LUV', 'JBHT', 'XPO', 'CHRW', 'UPS', 'FDX', 'KSU', 'MAR']
}

# 2. FUNÇÕES TÉCNICAS
def calculate_half_life(residue):
    delta_residue = residue.diff().dropna()
    lagged_residue = residue.shift(1).dropna()
    if len(lagged_residue) < 10: return 999
    x = sm.add_constant(lagged_residue.values)
    model = sm.OLS(delta_residue.values, x).fit()
    lambda_val = model.params[1]
    return -np.log(2) / lambda_val if lambda_val < 0 else 999

# 3. INTERFACE LATERAL
st.sidebar.header("📊 Parâmetros de Trading")
bp_limit = st.sidebar.number_input("Buying Power por Par ($)", value=1000)
z_threshold = st.sidebar.slider("Z-Score de Entrada", 1.0, 3.0, 1.96)
risk_target = st.sidebar.number_input("Risco Alvo (Volatilidade) $", value=15)
z_stop = st.sidebar.number_input("Z-Score de Stop (Emergência)", value=4.0)

# 4. EXECUÇÃO DO SCANNER
if st.sidebar.button("RODAR SCANNER AGORA"):
    all_tickers = [t for sub in market_universe.values() for t in sub]

    with st.spinner('Baixando dados do Yahoo Finance...'):
        raw_data = yf.download(all_tickers, period='350d', interval='1d', auto_adjust=True, progress=False)
        data = raw_data['Close'].dropna(axis=1)

    results = []
    windows = [100, 120, 140, 160, 180, 200, 220, 240, 250]

    progress_bar = st.progress(0)
    sectors = list(market_universe.keys())

    for i, sector in enumerate(sectors):
        available = [t for t in market_universe[sector] if t in data.columns]
        pairs = list(combinations(available, 2))

        for s1, s2 in pairs:
            try:
                coint_count = sum(1 for w in windows if coint(data[s1].tail(w), data[s2].tail(w))[1] < 0.05)
                if coint_count < 3: continue

                y, x = data[s1].tail(250).values, sm.add_constant(data[s2].tail(250).values)
                model = sm.OLS(y, x).fit()
                beta = model.params[1]
                if beta < 0.20: continue

                residue = data[s1].tail(250) - (model.params[0] + beta * data[s2].tail(250))
                z_score = (residue.iloc[-1] - residue.mean()) / residue.std()

                if abs(z_score) >= z_threshold and abs(z_score) < z_stop:
                    hl = calculate_half_life(residue)
                    if 1 <= hl <= 30:
                        p1, p2 = data[s1].iloc[-1], data[s2].iloc[-1]
                        vol_res = residue.std()

                        l1 = max(1, int(risk_target / (abs(z_score) * vol_res)))
                        l2 = max(1, int(l1 * beta))

                        while (l1 * p1 + l2 * p2) > bp_limit and l1 > 1:
                            l1 -= 1
                            l2 = max(1, int(l1 * beta))

                        custo_total = (l1 * p1 + l2 * p2)
                        if custo_total <= bp_limit:
                            acao = f"VENDE {s1}/COMPRA {s2}" if z_score > 0 else f"COMPRA {s1}/VENDE {s2}"
                            alvo_usd = abs(z_score) * vol_res * l1
                            stop_usd = (z_stop - abs(z_score)) * vol_res * l1

                            results.append({
                                "Setor": sector,
                                "Ação": acao,
                                "Lotes": f"{l1} / {l2}",
                                "Beta": round(beta, 2),
                                "Z": round(z_score, 2),
                                "Alvo($)": f"${alvo_usd:.2f}",
                                "Stop($)": f"-${stop_usd:.2f}",
                                "HL": round(hl, 1),
                                "Conf": f"{coint_count}/9"
                            })
            except: continue
        progress_bar.progress((i + 1) / len(sectors))

    if results:
        df_final = pd.DataFrame(results).sort_values(by='Z', ascending=False, key=abs)
        st.success(f"Foram encontradas {len(df_final)} oportunidades!")
        st.dataframe(df_final, use_container_width=True)
    else:
        st.warning("Nenhuma oportunidade encontrada com os filtros atuais.")

st.sidebar.markdown("---")
st.sidebar.info("Saia da operação quando o Z chegar em 0.00.")
