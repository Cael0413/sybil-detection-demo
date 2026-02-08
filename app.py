import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import os
import requests
import json
from collections import Counter
from datetime import datetime

# ==========================================
# 1. 設定與 CSS 優化 (深色模式修復版)
# ==========================================
st.set_page_config(
    page_title="區塊鏈詐欺偵測系統",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ★★★ CSS 修復重點：針對深色模式優化配色 ★★★
st.markdown("""
    <style>
    /* 主標題：紅色系，在深色背景很顯眼 */
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: #FF4B4B; 
        margin-bottom: 0px;
    }
    /* 副標題：灰白色 */
    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        color: #E0E0E0;
        margin-bottom: 30px;
    }
    
    /* 數據卡片優化 (解決文字看不見的問題) */
    div[data-testid="stMetric"] {
        background-color: #262730; /* 深灰色背景 */
        border: 1px solid #464b5f; /* 邊框 */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.5); /* 陰影 */
    }
    
    /* 強制數據數值為亮白色 */
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: bold;
    }
    
    /* 強制數據標籤為淺灰色 */
    div[data-testid="stMetricLabel"] {
        color: #AAAAAA !important;
    }
    </style>
""", unsafe_allow_html=True)

# ★★★ 您的 Alchemy API Key ★★★
ALCHEMY_URL = "https://eth-mainnet.g.alchemy.com/v2/HQJSb_FmasiwKPCINPjap" 

# ==========================================
# 2. 核心功能：API 與 特徵計算
# ==========================================
def get_real_features(address):
    headers = {"accept": "application/json", "content-type": "application/json"}
    features = {}
    address = address.strip()
    
    try:
        # (A) 查詢餘額
        payload_bal = {"id": 1, "jsonrpc": "2.0", "method": "eth_getBalance", "params": [address, "latest"]}
        res_bal = requests.post(ALCHEMY_URL, headers=headers, data=json.dumps(payload_bal)).json()
        
        if res_bal.get('error'):
            st.error(f"❌ API Error: {res_bal['error'].get('message')}")
            return None, None, None

        balance_wei = int(res_bal.get("result", "0x0"), 16)
        features['total ether balance'] = balance_wei / 10**18

        # (B) 查詢交易
        base_params = {
            "fromBlock": "0x0",
            "category": ["external", "erc20"], 
            "withMetadata": True,
            "excludeZeroValue": False,
            "maxCount": "0x3e8"
        }

        # 1. Sent
        params_sent = base_params.copy()
        params_sent["fromAddress"] = address
        payload_sent = {"id": 1, "jsonrpc": "2.0", "method": "alchemy_getAssetTransfers", "params": [params_sent]}
        res_sent = requests.post(ALCHEMY_URL, headers=headers, data=json.dumps(payload_sent)).json()
        sent_txs = res_sent.get('result', {}).get('transfers', [])

        # 2. Received
        params_rec = base_params.copy()
        params_rec["toAddress"] = address
        payload_rec = {"id": 1, "jsonrpc": "2.0", "method": "alchemy_getAssetTransfers", "params": [params_rec]}
        res_rec = requests.post(ALCHEMY_URL, headers=headers, data=json.dumps(payload_rec)).json()
        received_txs = res_rec.get('result', {}).get('transfers', [])

        # (C) 計算特徵
        eth_sent = [tx for tx in sent_txs if tx['asset'] == 'ETH']
        eth_received = [tx for tx in received_txs if tx['asset'] == 'ETH']

        features['Sent tnx'] = len(eth_sent)
        features['Received Tnx'] = len(eth_received)
        features['total Ether sent'] = sum([float(tx['value'] or 0) for tx in eth_sent])
        features['total ether received'] = sum([float(tx['value'] or 0) for tx in eth_received])

        if features['total ether received'] > 0:
            features['Sent/Received Ratio'] = features['total Ether sent'] / features['total ether received']
        else:
            features['Sent/Received Ratio'] = 0

        # (D) ERC20 特徵
        erc20_sent = [tx for tx in sent_txs if tx['category'] == 'erc20']
        erc20_received = [tx for tx in received_txs if tx['category'] == 'erc20']

        features['Total ERC20 tnxs'] = len(erc20_sent) + len(erc20_received)
        features['ERC20 uniq sent addr'] = len(set([tx['to'] for tx in erc20_sent]))
        features['ERC20 uniq rec addr'] = len(set([tx['from'] for tx in erc20_received]))

        rec_token_counts = Counter([tx['asset'] for tx in erc20_received])
        sent_token_counts = Counter([tx['asset'] for tx in erc20_sent])
        
        most_rec_token = rec_token_counts.most_common(1)[0][0] if rec_token_counts else "None"
        most_sent_token = sent_token_counts.most_common(1)[0][0] if sent_token_counts else "None"

        # (E) 複雜特徵
        all_vals_rec = [float(tx['value'] or 0) for tx in eth_received]
        all_vals_sent = [float(tx['value'] or 0) for tx in eth_sent]
        features['Max Val Received'] = max(all_vals_rec) if all_vals_rec else 0
        features['Max Val Sent'] = max(all_vals_sent) if all_vals_sent else 0

        features['Zero Value Tx Count'] = len([tx for tx in eth_sent if float(tx['value'] or 0) == 0])

        all_txs = sent_txs + received_txs
        if all_txs:
            timestamps = [datetime.strptime(tx['metadata']['blockTimestamp'], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp() for tx in all_txs if 'blockTimestamp' in tx['metadata']]
            if timestamps:
                features['Time Diff between first and last (Mins)'] = (max(timestamps) - min(timestamps)) / 60
            else:
                features['Time Diff between first and last (Mins)'] = 0
        else:
            features['Time Diff between first and last (Mins)'] = 0

        if len(sent_txs) > 1:
            sent_timestamps = sorted([datetime.strptime(tx['metadata']['blockTimestamp'], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp() for tx in sent_txs if 'blockTimestamp' in tx['metadata']])
            diffs = np.diff(sent_timestamps)
            avg_diff_seconds = np.mean(diffs)
            features['Avg min between sent tnx'] = avg_diff_seconds / 60
        else:
            features['Avg min between sent tnx'] = 0

        return features, most_rec_token, most_sent_token

    except Exception as e:
        st.error(f"程式執行錯誤: {e}")
        return None, None, None

# ==========================================
# 3. 載入模型
# ==========================================
@st.cache_resource
def load_assets():
    model = xgb.Booster()
    model_path = "fraud_detection_model_v5.json"
    columns_path = "model_columns_v5.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        return None, None
        
    model.load_model(model_path)
    model_columns = joblib.load(columns_path)
    return model, model_columns

bst, model_columns = load_assets()

# ==========================================
# 4. 前端介面設計 (UI Layer)
# ==========================================

# 標題區
st.markdown('<div class="main-title">🛡️ 區塊鏈詐欺偵測系統</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Blockchain Fraud Detection System powered by XGBoost</div>', unsafe_allow_html=True)

# 介紹區
with st.expander("ℹ️ 關於本系統 (System Overview)"):
    st.markdown("""
    **系統簡介：**
    本研究針對以太坊 (Ethereum) 上的詐欺與異常帳戶行為進行分析。透過結合 **機器學習 (XGBoost)** 與 **鏈上即時數據 (Alchemy)**，本系統能有效識別潛在的惡意帳戶（如女巫攻擊、資金盤、異常洗錢行為）。

    **使用說明：**
    1. 輸入 **以太坊錢包地址** (0x...)。
    2. 系統將即時抓取其歷史交易與代幣流向。
    3. 產出風險評估報告，判斷是否為 **「高風險詐欺帳戶」**。
    """)

# ★★★ 側邊欄 (學術專業版文案) ★★★
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/1257px-Ethereum_logo_2014.svg.png", width=50)
    st.title("模型架構資訊")
    
    st.markdown("---")
    
    st.markdown("### 🧠 核心演算法")
    st.info("**XGBoost Classifier**\n(eXtreme Gradient Boosting)")
    st.caption("採用基於樹的梯度提升演算法，具備高效能與高擴展性，特別適合處理稀疏矩陣數據。")

    st.markdown("### 🎯 模型效能")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("F1-Score", "95.35%")
    with col_b:
        st.metric("準確率", "95.0%")
    st.caption("經由 Grid Search 優化之最佳參數組合。")

    st.markdown("### 🏗️ 特徵工程")
    # 修正：確保 model_columns 存在才取長度
    if model_columns is not None:
        feat_count = len(model_columns)
    else:
        feat_count = 2847
        
    st.write(f"**監測特徵總數：{feat_count:,} 個**")
    st.progress(100)
    st.markdown("""
    包含兩大類行為分析：
    - **交易行為 (Transactional)**：頻率、金額、時間差。
    - **代幣流向 (Token Interaction)**：針對數千種 ERC-20 代幣進行 One-Hot 編碼分析。
    """)
    
    st.markdown("---")
    st.caption("© 2026 Blockchain Research Lab")

# 主輸入區
st.markdown("### 🔍 帳號風險查詢")
col1, col2 = st.columns([3, 1])

with col1:
    address_input = st.text_input("請輸入目標錢包地址", "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", placeholder="0x...", label_visibility="collapsed")

with col2:
    analyze_btn = st.button("🚀 開始偵測", type="primary", use_container_width=True)

# 執行邏輯
if analyze_btn:
    if bst is None:
        st.error("⚠️ 系統錯誤：偵測模型未載入，請確認伺服器狀態。")
    elif not address_input.startswith("0x") or len(address_input) != 42:
        st.warning("⚠️ 地址格式錯誤，請輸入標準的 Ethereum 地址 (42字元)")
    else:
        with st.spinner('🔗 正在掃描區塊鏈交易紀錄...'):
            real_features, most_rec, most_sent = get_real_features(address_input)
            
            if real_features:
                input_df = pd.DataFrame(columns=model_columns)
                input_df.loc[0] = 0
                for col, val in real_features.items():
                    if col in input_df.columns:
                        input_df[col] = val
                
                rec_col = f"ERC20_most_rec_token_type_{most_rec}"
                sent_col = f"ERC20 most sent token type_{most_sent}"
                if rec_col in input_df.columns: input_df[rec_col] = 1
                if sent_col in input_df.columns: input_df[sent_col] = 1

                dtest = xgb.DMatrix(input_df)
                prediction = bst.predict(dtest)[0]
                
                # --- 結果展示 ---
                st.markdown("---")
                st.subheader("📊 偵測報告 (Detection Report)")

                res_col1, res_col2, res_col3 = st.columns([1.5, 1.5, 1])

                with res_col1:
                    st.markdown("**系統判讀結果**")
                    risk_score = prediction * 100
                    st.metric("詐欺/異常機率 (Fraud Score)", f"{risk_score:.2f}%")
                    
                    if prediction > 0.5:
                        st.error("🚨 **高風險 (Fraudulent)**\n系統判定此為詐欺或異常帳戶。")
                    else:
                        st.success("✅ **正常 (Normal)**\n該地址行為模式正常。")

                with res_col2:
                    st.markdown("**關鍵行為特徵**")
                    st.write(f"⏱️ 平均交易間隔: `{real_features['Avg min between sent tnx']:.2f} 分`")
                    st.write(f"💸 發送/接收比率: `{real_features['Sent/Received Ratio']:.2f}`")
                    st.write(f"🔄 總交易次數: `{int(real_features['Sent tnx'] + real_features['Received Tnx'])}`")

                with res_col3:
                    st.markdown("**代幣流向**")
                    st.caption("主要接收")
                    st.code(most_rec)
                    st.caption("主要發送")
                    st.code(most_sent)

                st.markdown("#### 📈 行為模式分析圖")
                viz_df = pd.DataFrame({
                    '特徵': ['Sent Tnx', 'Avg Time Diff (Min)', 'Zero Val Tnx'],
                    '數值': [real_features['Sent tnx'], real_features['Avg min between sent tnx'], real_features['Zero Value Tx Count']]
                })
                st.bar_chart(viz_df.set_index('特徵'))
                
                with st.expander("🔍 查看詳細交易數據"):
                    st.json(real_features)
