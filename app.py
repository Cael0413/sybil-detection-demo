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
# 1. 設定與初始化
# ==========================================
st.set_page_config(page_title="區塊鏈詐騙偵測系統", page_icon="🛡️", layout="wide")

# ★★★ 您的 Alchemy API Key ★★★
ALCHEMY_URL = "https://eth-mainnet.g.alchemy.com/v2/HQJSb_FmasiwKPCINPjap" 

st.title("🛡️ 區塊鏈女巫攻擊偵測系統")
st.markdown("### 基於 XGBoost 與 Alchemy 鏈上實時分析")

# ==========================================
# 2. 核心功能：從區塊鏈抓取並計算 (已修復 Hex 格式錯誤)
# ==========================================
def get_real_features(address):
    headers = {"accept": "application/json", "content-type": "application/json"}
    features = {}
    
    # 去除前後空白
    address = address.strip()
    
    try:
        # --- (A) 查詢 ETH 餘額 ---
        payload_bal = {"id": 1, "jsonrpc": "2.0", "method": "eth_getBalance", "params": [address, "latest"]}
        res_bal = requests.post(ALCHEMY_URL, headers=headers, data=json.dumps(payload_bal)).json()
        
        if res_bal.get('error'):
            st.error(f"❌ 餘額查詢失敗: {res_bal['error'].get('message')}")
            return None, None, None

        balance_wei = int(res_bal.get("result", "0x0"), 16)
        features['total ether balance'] = balance_wei / 10**18

        # --- (B) 查詢交易紀錄 ---
        # 參數修正：maxCount 必須是 16 進位字串 (1000 -> "0x3e8")
        base_params = {
            "fromBlock": "0x0",
            "category": ["external", "erc20"], 
            "withMetadata": True,
            "excludeZeroValue": False,
            "maxCount": "0x3e8"  # <--- 【關鍵修正】改成 16 進位格式！
        }

        # 1. 抓取「發送」紀錄
        params_sent = base_params.copy()
        params_sent["fromAddress"] = address
        payload_sent = {"id": 1, "jsonrpc": "2.0", "method": "alchemy_getAssetTransfers", "params": [params_sent]}
        
        res_sent = requests.post(ALCHEMY_URL, headers=headers, data=json.dumps(payload_sent)).json()
        
        if res_sent.get('error'):
            st.error(f"❌ 交易紀錄查詢失敗 (Sent): {res_sent['error'].get('message')}")
            st.code(json.dumps(res_sent, indent=2))
            return None, None, None
            
        sent_txs = res_sent.get('result', {}).get('transfers', [])

        # 2. 抓取「接收」紀錄
        params_rec = base_params.copy()
        params_rec["toAddress"] = address
        payload_rec = {"id": 1, "jsonrpc": "2.0", "method": "alchemy_getAssetTransfers", "params": [params_rec]}
        
        res_rec = requests.post(ALCHEMY_URL, headers=headers, data=json.dumps(payload_rec)).json()
        
        if res_rec.get('error'):
            st.error(f"❌ 交易紀錄查詢失敗 (Received): {res_rec['error'].get('message')}")
            return None, None, None
            
        received_txs = res_rec.get('result', {}).get('transfers', [])

        # --- (C) 計算 ETH 相關特徵 ---
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

        # --- (D) 計算 ERC20 相關特徵 ---
        erc20_sent = [tx for tx in sent_txs if tx['category'] == 'erc20']
        erc20_received = [tx for tx in received_txs if tx['category'] == 'erc20']

        features['Total ERC20 tnxs'] = len(erc20_sent) + len(erc20_received)
        features['ERC20 uniq sent addr'] = len(set([tx['to'] for tx in erc20_sent]))
        features['ERC20 uniq rec addr'] = len(set([tx['from'] for tx in erc20_received]))

        rec_token_counts = Counter([tx['asset'] for tx in erc20_received])
        sent_token_counts = Counter([tx['asset'] for tx in erc20_sent])
        
        most_rec_token = rec_token_counts.most_common(1)[0][0] if rec_token_counts else "None"
        most_sent_token = sent_token_counts.most_common(1)[0][0] if sent_token_counts else "None"

        # --- (E) 計算複雜特徵 ---
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
# 3. 載入模型與欄位
# ==========================================
@st.cache_resource
def load_assets():
    model = xgb.Booster()
    model_path = "fraud_detection_model_v5.json"
    columns_path = "model_columns_v5.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        st.error("❌ 找不到模型檔案，請先執行 train_optimized.py")
        return None, None
        
    model.load_model(model_path)
    model_columns = joblib.load(columns_path)
    return model, model_columns

bst, model_columns = load_assets()

# ==========================================
# 4. 前端介面邏輯
# ==========================================
with st.sidebar:
    st.header("關於模型")
    st.info("本系統採用 XGBoost Embedded Method 篩選特徵，準確率達 95%。")
    if model_columns is not None:
        st.write(f"模型特徵總數: {len(model_columns)} 個")

address_input = st.text_input("請輸入以太坊錢包地址 (0x...)", "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045") 

if st.button("開始實時分析", type="primary"):
    if bst is None:
        st.error("無法執行分析：模型未載入")
    elif not address_input.startswith("0x") or len(address_input) != 42:
        st.warning("⚠️ 地址格式錯誤")
    else:
        with st.spinner('🔗 正在連線 Alchemy 區塊鏈節點...'):
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
                
                st.divider()
                col1, col2, col3 = st.columns([1, 2, 2])
                with col1:
                    st.metric("詐騙機率", f"{prediction:.2%}")
                    if prediction > 0.5:
                        st.error("🚨 高風險 (Sybil)")
                    else:
                        st.success("✅ 安全 (Normal)")
                with col2:
                    st.subheader("📊 關鍵行為數值")
                    st.write(f"⏱️ 平均發送間隔: **{real_features['Avg min between sent tnx']:.2f} 分**")
                    st.write(f"💸 發送/接收比率: **{real_features['Sent/Received Ratio']:.2f}**")
                    st.write(f"🔄 總交易次數: **{real_features['Sent tnx'] + real_features['Received Tnx']}**")
                with col3:
                    st.subheader("🪙 代幣偏好")
                    st.write(f"📥 最常接收: **{most_rec}**")
                    st.write(f"📤 最常發送: **{most_sent}**")
                
                viz_cols = ['Sent tnx', 'Avg min between sent tnx', 'Zero Value Tx Count']
                st.bar_chart(input_df[viz_cols].T)
                
                with st.expander("查看原始 JSON 數據 (驗證用)"):
                    st.json(real_features)