# tpl-rules-agent-gemini（TPL 規則代理人，Gemini 版）

`POST /v1/tpl/rules`：融合檢索（Chroma＋BM25＋GraphRAG）＋ Gemini（prompt v5）判斷
理賠／拒賠／資料不足／疑似詐欺。`GET /health` 健康檢查（不要用 `/healthz`，Cloud Run 會擋）。

2026-11-03 起支援法官退回重算：請求可帶選填欄位 `judge_feedback`（法官的退回理由）與
`recalc_round`（第幾次退回，1–3）。檢索仍只用原本的 `case_description`，退回理由附在 prompt 後面。
沒帶這兩個欄位時行為與先前完全相同。

2026-11-25 起為 **v8**（01 v8 notebook）：
- 選填 `record_counts`：`{"episode_count", "seq_count", "visit_count", "injury_insured"}`
  （claim_episodes／coverages／medical_visits 筆數、31／33 險種保額合計，都不去重）。API 依規則 A／B
  算出 `[詐欺偵測規則]` 區塊附在案件描述最後（只給 Gemini，不進檢索），回應多 `fraud_rule`。
  呼叫端也可以自己把區塊附在 `case_description` 最後，API 會在檢索前把它拆掉。
- 檢索到 train 的 Isolation Forest 離群案例（`app/data/train_if_outliers.txt`，一行一個案號）時，
  內文前面加 `[異常偵測標記：疑似詐欺]`；回應多 `n_fraud_flagged_claims`。
- system prompt 預設 v8；環境變數 `PROMPT_VERSION=v5` 可回退。`FRAUD_FLAG_ENABLED=0`／`FRAUD_RULE_ENABLED=0` 可關閉。

## 檔案

| 檔案 | 內容 |
|---|---|
| `app/retrieval.py` | 01 v6 融合檢索、證據排版、輸出正規化（逐字沿用組員版本，請勿修改） |
| `app/gemini_agent.py` | `call_rules_agent()`、system prompt v8（`PROMPT_VERSION=v5` 可回退） |
| `app/fraud.py` | v8 詐欺偵測規則 A／B、IF 離群標記 |
| `app/data/train_if_outliers.txt` | train 36 件 IF 離群案號（**不可放 val**） |
| `app/main.py` | FastAPI；`judge_feedback` 處理在 `_with_judge_feedback()` |
| `Dockerfile` | 以線上映像 `tpl-rules/api:gemini-v5` 為底，只覆蓋 `app/` |

模型與資料（E5、ChromaDB、BM25 DB）不放在這個 repo，來自底層映像的 `/app/artifacts/`；
原始檔在 `gs://rsa-cki-scu-2026/TPL_RAG_DB/`。

## 環境變數（設定在 Cloud Run 服務上，不在程式碼裡）

| 變數 | 來源 |
|---|---|
| `API_KEY` | Secret `tpl-rules-agent-v2-key` |
| `GEMINI_API_KEY` | Secret `gemini-api-key-rules` |
| `GRAPHRAG_API_KEY` | Secret `graphrag-api-key-v2` |

## 部署

push 到 `main` → Cloud Build 觸發器（asia-southeast1）依 `cloudbuild.yaml` 建映像、推到
`tpl-rules/api:<commit SHA>`、`gcloud run deploy` 更新 `tpl-rules-agent-gemini`。只換映像，
服務原有的環境變數、Secret、服務帳號、CPU／記憶體設定都保留。
