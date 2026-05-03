import os
import json
from datetime import datetime

import joblib
import pandas as pd
import requests
import streamlit as st


# ==========================================
# 1. 基本設定
# ==========================================
st.set_page_config(
    page_title="區塊鏈詐騙錢包偵測系統",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.7rem;
        font-weight: 700;
        color: #FF4B4B;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #D9D9D9;
        margin-bottom: 24px;
    }
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #464b5f;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.35);
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        color: #BBBBBB !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. API 金鑰與鏈別設定
# ==========================================
def get_api_key():
    try:
        if "ALCHEMY_API_KEY" in st.secrets:
            return st.secrets["ALCHEMY_API_KEY"]
    except Exception:
        pass
    return os.getenv("ALCHEMY_API_KEY")


API_KEY = "HQJSb_FmasiwKPCINPjap"

NETWORK_URLS = {}
if API_KEY:
    NETWORK_URLS = {
        "Ethereum (以太坊主網)": f"https://eth-mainnet.g.alchemy.com/v2/{API_KEY}",
        "Arbitrum (L2)": f"https://arb-mainnet.g.alchemy.com/v2/{API_KEY}",
        "Polygon (Matic)": f"https://polygon-mainnet.g.alchemy.com/v2/{API_KEY}",
        "Base (L2)": f"https://base-mainnet.g.alchemy.com/v2/{API_KEY}",
        "Optimism (L2)": f"https://opt-mainnet.g.alchemy.com/v2/{API_KEY}",
        "BNB Chain (BSC)": f"https://bnb-mainnet.g.alchemy.com/v2/{API_KEY}",
    }

# 7 天視窗估計區塊數
BLOCK_WINDOW = 50400


# ==========================================
# 3. 小工具函式
# ==========================================
def safe_post(url, payload):
    headers = {"accept": "application/json", "content-type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
    response.raise_for_status()
    return response.json()


def is_valid_evm_address(address):
    return address.startswith("0x") and len(address) == 42


def build_column_lookup(model_columns):
    lookup = {}
    for col in model_columns:
        lookup[col.strip()] = col
    return lookup


def align_features_to_model_columns(real_features, model_columns):
    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0

    column_lookup = build_column_lookup(model_columns)

    for raw_key, value in real_features.items():
        normalized_key = raw_key.strip()
        if normalized_key in column_lookup:
            input_df.at[0, column_lookup[normalized_key]] = value

    return input_df


# ==========================================
# 4. 特徵擷取
# ==========================================
def get_real_features(address, alchemy_url):
    features = {}
    all_timestamps = []

    try:
        address = address.strip()

        # A. 找到首次收到資金之區塊
        payload_first = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "toAddress": address,
                    "category": ["external", "internal", "erc20"],
                    "order": "asc",
                    "maxCount": "0x1",
                }
            ],
        }
        res_first = safe_post(alchemy_url, payload_first)
        transfers_first = res_first.get("result", {}).get("transfers", [])

        if not transfers_first:
            return None, "此地址在該網路上為空錢包或尚無交易紀錄。"

        first_block_hex = transfers_first[0]["blockNum"]
        first_block_int = int(first_block_hex, 16)
        end_block_hex = hex(first_block_int + BLOCK_WINDOW)

        base_params = {
            "fromBlock": first_block_hex,
            "toBlock": end_block_hex,
            "withMetadata": True,
            "maxCount": "0x3e8",
        }

        # 1. 原生代幣接收
        payload_in = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [{**base_params, "toAddress": address, "category": ["external"]}],
        }
        txs_in = safe_post(alchemy_url, payload_in).get("result", {}).get("transfers", [])

        features["Received Tnx"] = len(txs_in)
        vals_in = [float(tx.get("value") or 0) for tx in txs_in]
        features["total ether received"] = sum(vals_in)
        features["Max Val Received"] = max(vals_in) if vals_in else 0

        for tx in txs_in:
            if "metadata" in tx and "blockTimestamp" in tx["metadata"]:
                all_timestamps.append(tx["metadata"]["blockTimestamp"])

        # 2. 原生代幣發送
        payload_out = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [{**base_params, "fromAddress": address, "category": ["external"]}],
        }
        txs_out = safe_post(alchemy_url, payload_out).get("result", {}).get("transfers", [])

        features["Sent tnx"] = len(txs_out)
        vals_out = [float(tx.get("value") or 0) for tx in txs_out]
        features["total Ether sent"] = sum(vals_out)
        features["Max Val Sent"] = max(vals_out) if vals_out else 0

        for tx in txs_out:
            if "metadata" in tx and "blockTimestamp" in tx["metadata"]:
                all_timestamps.append(tx["metadata"]["blockTimestamp"])

        # 3. ERC20 接收 / 發送
        payload_in_20 = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [{**base_params, "toAddress": address, "category": ["erc20"]}],
        }
        txs_in_20 = safe_post(alchemy_url, payload_in_20).get("result", {}).get("transfers", [])

        payload_out_20 = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [{**base_params, "fromAddress": address, "category": ["erc20"]}],
        }
        txs_out_20 = safe_post(alchemy_url, payload_out_20).get("result", {}).get("transfers", [])

        features["Total ERC20 tnxs"] = len(txs_in_20) + len(txs_out_20)
        features["ERC20 uniq rec addr"] = float(len(set(tx.get("from") for tx in txs_in_20 if tx.get("from"))))
        features["ERC20 uniq sent addr"] = float(len(set(tx.get("to") for tx in txs_out_20 if tx.get("to"))))

        # 4. 零值交易
        zero_count = 0
        for tx in txs_out + txs_out_20:
            tx_val = tx.get("value")
            if tx_val == 0 or tx_val == "0" or tx_val is None:
                zero_count += 1
        features["Zero Value Tx Count"] = zero_count

        # 5. 收送比例
        total_received = features["Received Tnx"] + len(txs_in_20)
        total_sent = features["Sent tnx"] + len(txs_out_20)
        features["Sent/Received Ratio"] = (total_sent / total_received) if total_received > 0 else 0

        # 6. 時間特徵
        if all_timestamps:
            ts_list = [datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() for t in all_timestamps]
            ts_list.sort()
            features["Time Diff between first and last (Mins)"] = (ts_list[-1] - ts_list[0]) / 60

            if len(ts_list) > 1:
                features["Avg min between sent tnx"] = ((ts_list[-1] - ts_list[0]) / (len(ts_list) - 1)) / 60
            else:
                features["Avg min between sent tnx"] = 0
        else:
            features["Time Diff between first and last (Mins)"] = 0
            features["Avg min between sent tnx"] = 0

        return features, None

    except requests.exceptions.RequestException as e:
        return None, f"API 連線失敗：{e}"
    except Exception as e:
        return None, f"特徵擷取失敗：{e}"


# ==========================================
# 5. 載入模型
# ==========================================
@st.cache_resource
def load_assets():
    model_path = "fraud_detector_xgb_t7.joblib"
    columns_path = "model_columns_v5.joblib"

    if not os.path.exists(model_path):
        return None, None, f"找不到模型檔：{model_path}"
    if not os.path.exists(columns_path):
        return None, None, f"找不到欄位檔：{columns_path}"

    try:
        model = joblib.load(model_path)
        model_columns = joblib.load(columns_path)
        return model, model_columns, None
    except Exception as e:
        return None, None, f"模型載入失敗：{e}"


model, model_columns, load_error = load_assets()


# ==========================================
# 6. 前端介面
# ==========================================
st.markdown('<div class="main-title">🛡️ 區塊鏈詐騙錢包偵測系統</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">XGBoost + Dynamic Time-Window Features + Rule-based Safety Layer</div>',
    unsafe_allow_html=True,
)

with st.expander("ℹ️ 關於本系統"):
    st.markdown(
        """
        **本系統為研究展示與概念驗證原型。**

        主要設計理念如下：

        1. **動態時間窗特徵工程（Dynamic Time-Window）**  
           以地址首次收到資金作為共同起點，擷取其後固定視窗內之可觀察鏈上行為。

        2. **XGBoost 主模型**  
           作為主要風險分類引擎，用於辨識可疑詐騙錢包地址。

        3. **規則式安全防線（Rule-based Safety Layer）**  
           針對極端異常之大額高頻行為，提供額外風險覆蓋機制。

        4. **多鏈雷達展示（Multi-chain Radar）**  
           支援多個 EVM 相容網路的地址掃描與早期風險檢測展示。
        """
    )

with st.sidebar:
    st.title("模型資訊")

    st.markdown("### 🧠 核心模型")
    st.info("**XGBoost 主模型**\n\nT+7 Dynamic Time-Window Setting")

    st.markdown("### 📊 主要效能")
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("ROC-AUC", "94.32%")
    with metric_col2:
        st.metric("PR-AUC", "91.98%")

    metric_col3, metric_col4 = st.columns(2)
    with metric_col3:
        st.metric("F1-score", "83.99%")
    with metric_col4:
        st.metric("Recall", "84.62%")

    st.caption("以上指標對應論文 T+7 主模型測試結果。")

    st.markdown("---")
    st.markdown("### ⚙️ 系統說明")
    st.caption("本系統為研究展示用途，結果應作為輔助風險判讀參考。")

    st.markdown("---")
    st.caption("© 2026 Blockchain Research Lab")

st.markdown("### 🔍 地址首週行為分析")

if not API_KEY:
    st.error("未讀取到 ALCHEMY_API_KEY，請先於 Streamlit secrets 或環境變數中設定。")

if load_error:
    st.error(load_error)

col_net, col_addr = st.columns([1, 2.5])

with col_net:
    selected_network_name = st.selectbox(
        "🌐 掃描網路",
        list(NETWORK_URLS.keys()) if NETWORK_URLS else ["尚未設定 API Key"],
        disabled=not bool(NETWORK_URLS),
    )

with col_addr:
    address_input = st.text_input(
        "請輸入目標錢包地址",
        value="0x59ABf3837Fa962d6853b4Cc0a19513AA031fd32b",
        placeholder="0x...",
    )

analyze_btn = st.button("🚀 啟動偵測", type="primary", use_container_width=True)

if analyze_btn:
    if not API_KEY:
        st.error("請先設定 ALCHEMY_API_KEY。")
    elif model is None or model_columns is None:
        st.error("模型尚未成功載入，請確認模型檔與欄位檔是否正確。")
    elif not is_valid_evm_address(address_input):
        st.warning("地址格式錯誤，請輸入標準 EVM 地址（42 字元）。")
    else:
        current_alchemy_url = NETWORK_URLS[selected_network_name]

        with st.spinner(f"正在 {selected_network_name} 上擷取首週行為資料並進行風險分析..."):
            real_features, error_msg = get_real_features(address_input, current_alchemy_url)

        if error_msg:
            st.warning(error_msg)
        elif real_features:
            try:
                input_df = align_features_to_model_columns(real_features, model_columns)
                prediction = float(model.predict_proba(input_df)[0][1])

                # ==========================================
                # 規則式安全防線
                # ==========================================
                is_extreme_anomaly = False
                total_txns = int(real_features.get("Sent tnx", 0) + real_features.get("Received Tnx", 0))

                if real_features.get("Max Val Received", 0) > 1000 and total_txns > 50:
                    is_extreme_anomaly = True
                    prediction = 0.9999

                risk_score = prediction * 100

                st.markdown("---")
                st.subheader("📊 偵測報告")

                res_col1, res_col2 = st.columns([1.5, 2])

                with res_col1:
                    st.markdown("**系統判讀結果**")
                    st.metric("風險分數（Risk Score）", f"{risk_score:.2f}%")

                    if is_extreme_anomaly:
                        st.error(
                            "🚨 **極高風險異常地址**\n\n"
                            "系統偵測到大額且高頻之異常鏈上行為，已觸發規則式安全防線。"
                        )
                    elif prediction > 0.5:
                        st.error("🚨 **高風險（Fraudulent）**\n\n模型判定該地址首週行為側寫具有詐騙風險。")
                    else:
                        st.success("✅ **低風險（Normal）**\n\n模型判定該地址首週行為模式相對正常。")

                with res_col2:
                    st.markdown("**首週關鍵行為摘要**")
                    st.write(f"⏱️ 平均交易間隔：`{real_features.get('Avg min between sent tnx', 0):.2f} 分`")
                    st.write(f"💸 發送 / 接收比率：`{real_features.get('Sent/Received Ratio', 0):.2f}`")
                    st.write(f"🔄 首週交易總數：`{total_txns} 次`")
                    st.write(f"💰 最大單筆接收：`{real_features.get('Max Val Received', 0):.4f}`")

                st.markdown("#### 📈 行為概覽圖")
                viz_df = pd.DataFrame(
                    {
                        "特徵": ["Sent Tnx", "Avg Time Diff (Min)", "Zero Value Tx Count"],
                        "數值": [
                            real_features.get("Sent tnx", 0),
                            real_features.get("Avg min between sent tnx", 0),
                            real_features.get("Zero Value Tx Count", 0),
                        ],
                    }
                )
                st.bar_chart(viz_df.set_index("特徵"))

                with st.expander("🔍 查看詳細萃取特徵"):
                    st.json(real_features)

                with st.expander("🧾 模型輸入欄位預覽"):
                    st.dataframe(input_df.T.rename(columns={0: "value"}))

            except Exception as e:
                st.error(f"模型推論失敗：{e}")
