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
# 1. 設定與 CSS 優化 (深色模式)
# ==========================================
st.set_page_config(
    page_title="區塊鏈早期防詐預警系統",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: #FF4B4B; 
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        color: #E0E0E0;
        margin-bottom: 30px;
    }
    div[data-testid="stMetric"] {
        background-color: #262730; 
        border: 1px solid #464b5f; 
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.5); 
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        color: #AAAAAA !important;
    }
    </style>
""", unsafe_allow_html=True)

# ★★★ 多鏈雷達 API 設定 (使用您的金鑰) ★★★
API_KEY = "HQJSb_FmasiwKPCINPjap"
NETWORK_URLS = {
    "Ethereum (以太坊主網)": f"https://eth-mainnet.g.alchemy.com/v2/{API_KEY}",
    "Arbitrum (L2)": f"https://arb-mainnet.g.alchemy.com/v2/{API_KEY}",
    "Polygon (Matic)": f"https://polygon-mainnet.g.alchemy.com/v2/{API_KEY}",
    "Base (L2)": f"https://base-mainnet.g.alchemy.com/v2/{API_KEY}",
    "Optimism (L2)": f"https://opt-mainnet.g.alchemy.com/v2/{API_KEY}",
    "BNB Chain (BSC)": f"https://bnb-mainnet.g.alchemy.com/v2/{API_KEY}"
}

BLOCK_WINDOW = 50400 # 7 天的區塊數量 (以太坊基準，其他鏈暫時共用此區塊數做基準)

# ==========================================
# 2. 核心功能：API 與 特徵計算 (加入指定網路 URL)
# ==========================================
def get_real_features(address, alchemy_url):
    headers = {"accept": "application/json", "content-type": "application/json"}
    features = {}
    all_timestamps = []
    
    try:
        address = address.strip()
        
        # 步驟 A：錢包起點
        payload_first = {
            "id": 1, "jsonrpc": "2.0", "method": "alchemy_getAssetTransfers",
            "params": [{"toAddress": address, "category": ["external", "internal", "erc20"], "order": "asc", "maxCount": "0x1"}]
        }
        res_first = requests.post(alchemy_url, headers=headers, data=json.dumps(payload_first)).json()
        transfers_first = res_first.get('result', {}).get('transfers', [])

        if not transfers_first:
            return None, "此地址在該網路上為空錢包或尚無交易紀錄"

        first_block_hex = transfers_first[0]['blockNum']
        first_block_int = int(first_block_hex, 16)
        end_block_hex = hex(first_block_int + BLOCK_WINDOW)

        # 步驟 B：在時間內擷取特徵
        params = {
            "fromBlock": first_block_hex, 
            "toBlock": end_block_hex, 
            "withMetadata": True, 
            "maxCount": "0x3e8"
        }

        # 1. 外部原生代幣接收
        payload_in = {"id": 1, "jsonrpc": "2.0", "method": "alchemy_getAssetTransfers", "params": [{**params, "toAddress": address, "category": ["external"]}]}
        txs_in = requests.post(alchemy_url, headers=headers, data=json.dumps(payload_in)).json().get('result', {}).get('transfers', [])
        
        features['Received Tnx'] = len(txs_in)
        vals_in = [float(x['value'] or 0) for x in txs_in]
        features['total ether received'] = sum(vals_in)
        features['Max Val Received'] = max(vals_in) if vals_in else 0
        for x in txs_in: 
            if 'metadata' in x: all_timestamps.append(x['metadata']['blockTimestamp'])

        # 2. 外部原生代幣發送
        payload_out = {"id": 1, "jsonrpc": "2.0", "method": "alchemy_getAssetTransfers", "params": [{**params, "fromAddress": address, "category": ["external"]}]}
        txs_out = requests.post(alchemy_url, headers=headers, data=json.dumps(payload_out)).json().get('result', {}).get('transfers', [])
        
        features['Sent tnx'] = len(txs_out)
        vals_out = [float(x['value'] or 0) for x in txs_out]
        features['total Ether sent'] = sum(vals_out)
        features['Max Val Sent'] = max(vals_out) if vals_out else 0
        for x in txs_out: 
            if 'metadata' in x: all_timestamps.append(x['metadata']['blockTimestamp'])

        # 3. ERC20 收發
        payload_in_20 = {"id": 1, "jsonrpc": "2.0", "method": "alchemy_getAssetTransfers", "params": [{**params, "toAddress": address, "category": ["erc20"]}]}
        txs_in_20 = requests.post(alchemy_url, headers=headers, data=json.dumps(payload_in_20)).json().get('result', {}).get('transfers', [])
        
        payload_out_20 = {"id": 1, "jsonrpc": "2.0", "method": "alchemy_getAssetTransfers", "params": [{**params, "fromAddress": address, "category": ["erc20"]}]}
        txs_out_20 = requests.post(alchemy_url, headers=headers, data=json.dumps(payload_out_20)).json().get('result', {}).get('transfers', [])

        features[' Total ERC20 tnxs'] = len(txs_in_20) + len(txs_out_20)
        features[' ERC20 uniq rec addr'] = float(len(set([x['from'] for x in txs_in_20])))
        features[' ERC20 uniq sent addr'] = float(len(set([x['to'] for x in txs_out_20])))

        # 4. 衍生特徵
        zero_count = 0
        for tx in txs_out + txs_out_20:
            if tx.get('value') == 0 or tx.get('value') is None:
                zero_count += 1
        features['Zero Value Tx Count'] = zero_count

        if features['Received Tnx'] + features[' Total ERC20 tnxs'] > 0:
            features['Sent/Received Ratio'] = (features['Sent tnx'] + len(txs_out_20)) / (features['Received Tnx'] + features[' Total ERC20 tnxs'])
        else:
            features['Sent/Received Ratio'] = 0

        if all_timestamps:
            ts_list = [datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp() for t in all_timestamps]
            ts_list.sort()
            features['Time Diff between first and last (Mins)'] = (ts_list[-1] - ts_list[0]) / 60
            if len(ts_list) > 1:
                features['Avg min between sent tnx'] = ((ts_list[-1] - ts_list[0]) / (len(ts_list)-1)) / 60
            else:
                features['Avg min between sent tnx'] = 0
        else:
            features['Time Diff between first and last (Mins)'] = 0
            features['Avg min between sent tnx'] = 0

        return features, None

    except Exception as e:
        return None, str(e)

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

st.markdown('<div class="main-title">🛡️ 區塊鏈防詐預警系統</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hybrid Architecture: ML Time-Windowing + Expert Rules + Multi-Chain Radar</div>', unsafe_allow_html=True)

with st.expander("ℹ️ 關於本系統架構 (System Architecture)"):
    st.markdown("""
    **【混合式防禦雙擎架構】**
    本系統不僅採用機器學習，更結合了專家防線，打造多維度的預警網：
    1. **動態時間窗 (Time Window)：** 嚴格擷取目標誕生後 7 天內的行為，根除生命週期外洩偏差。
    2. **規則引擎 (Rule-based Engine)：** 針對超越 AI 歷史認知極限的國家級/機構級巨鯨駭客（Out-of-Distribution 異常），啟動強制攔截防線。
    3. **多鏈雷達 (Multi-Chain Radar)：** 支援動態切換以太坊及各大 Layer-2 網路，讓跨鏈流竄的惡意空錢包無所遁形。
    """)

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/1257px-Ethereum_logo_2014.svg.png", width=50)
    st.title("模型架構資訊")
    
    st.markdown("---")
    
    st.markdown("### 🧠 核心技術")
    st.info("**XGBoost Classifier**\n+ Rule-based Override")
    st.caption("結合行為模型與領域專家規則，突破單一 AI 黑盒子極限。")

    st.markdown("### 🎯 嚴謹驗證效能")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("F1-Score", "80.55%")
    with col_b:
        st.metric("召回率 (Recall)", "87.0%")
    st.caption("經由 5-Fold 交叉驗證與特徵消融實驗成績。")
    
    st.markdown("---")
    st.caption("© 2026 Blockchain Research Lab")

st.markdown("### 🔍 帳號首週行為分析")

# 🌟 新增：多鏈雷達選擇器
col_net, col_addr = st.columns([1, 2.5])
with col_net:
    selected_network_name = st.selectbox("🌐 掃描網路 (多鏈雷達)", list(NETWORK_URLS.keys()))
with col_addr:
    address_input = st.text_input("請輸入目標錢包地址", "0x59ABf3837Fa962d6853b4Cc0a19513AA031fd32b", placeholder="0x...")

analyze_btn = st.button("🚀 啟動混合雙擎偵測", type="primary", use_container_width=True)

if analyze_btn:
    if bst is None:
        st.error("⚠️ 系統錯誤：偵測模型未載入，請確認 V5 版 JSON 與 Joblib 檔案是否在同一目錄下。")
    elif not address_input.startswith("0x") or len(address_input) != 42:
        st.warning("⚠️ 地址格式錯誤，請輸入標準的 EVM 地址 (42字元)")
    else:
        # 動態取得選擇的網路 URL
        current_alchemy_url = NETWORK_URLS[selected_network_name]
        
        with st.spinner(f'🔗 正在 {selected_network_name} 上尋找帳號起點並掃描交易...'):
            real_features, error_msg = get_real_features(address_input, current_alchemy_url)
            
            if error_msg:
                st.warning(f"⚠️ {error_msg}")
            elif real_features:
                input_df = pd.DataFrame(columns=model_columns)
                input_df.loc[0] = 0
                for col, val in real_features.items():
                    if col in input_df.columns:
                        input_df[col] = val
                
                # 將 DataFrame 轉換為 DMatrix 以供 XGBoost 預測
                dtest = xgb.DMatrix(input_df)
                prediction = bst.predict(dtest)[0]
                
                # ==========================================
                # 新增：規則防線 (Rule-based Override)
                # ==========================================
                is_super_whale_hacker = False
                total_txns = real_features['Sent tnx'] + real_features['Received Tnx']
                
                # 規則：首週接收超過 1000 顆原生代幣，且交易極度高頻 (>50次)
                if real_features['Max Val Received'] > 1000 and total_txns > 50:
                    is_super_whale_hacker = True
                    prediction = 0.9999 # 強制覆蓋 AI 預測
                
                # --- 結果展示 ---
                st.markdown("---")
                st.subheader("📊 偵測報告 (Detection Report)")

                res_col1, res_col2 = st.columns([1.5, 2])

                with res_col1:
                    st.markdown("**系統判讀結果**")
                    risk_score = prediction * 100
                    st.metric("詐欺/異常機率 (Fraud Score)", f"{risk_score:.2f}%")
                    
                    if is_super_whale_hacker:
                        st.error("🚨 **【重大安全警報】極端異常巨鯨**\n系統偵測到機構級巨量資金與極端高頻交易，觸發防護規則 (Rule-based Override)，判定為極高風險帳戶！")
                    elif prediction > 0.5:
                        st.error("🚨 **高風險 (Fraudulent)**\n系統 AI 模型判定其首週行為側寫符合詐欺特徵。")
                    else:
                        st.success("✅ **正常 (Normal)**\n該地址初期行為模式正常。")

                with res_col2:
                    st.markdown("**首週關鍵行為側寫**")
                    st.write(f"⏱️ 平均交易間隔: `{real_features['Avg min between sent tnx']:.2f} 分`")
                    st.write(f"💸 發送/接收比率: `{real_features['Sent/Received Ratio']:.2f}`")
                    st.write(f"🔄 首週交易總數: `{int(total_txns)} 次`")
                    st.write(f"💰 最大單筆接收: `{real_features['Max Val Received']:.4f}`")

                st.markdown("#### 📈 行為模式分析圖")
                viz_df = pd.DataFrame({
                    '特徵': ['Sent Tnx', 'Avg Time Diff (Min)', 'Zero Val Tnx'],
                    '數值': [real_features['Sent tnx'], real_features['Avg min between sent tnx'], real_features['Zero Value Tx Count']]
                })
                st.bar_chart(viz_df.set_index('特徵'))
                
                with st.expander("🔍 查看詳細萃取數據 (Debug)"):
                    st.json(real_features)
