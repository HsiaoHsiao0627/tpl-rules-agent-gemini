# tpl-rules-agent-gemini（TPL 規則代理人，Gemini 版）

`POST /v1/tpl/rules`：融合檢索（Chroma＋BM25＋GraphRAG）＋ Gemini（prompt v5）判斷
理賠／拒賠／資料不足／疑似詐欺。`GET /health` 健康檢查（不要用 `/healthz`，Cloud Run 會擋）。

2026-11-03 起支援法官退回重算：請求可帶選填欄位 `judge_feedback`（法官的退回理由）與
`recalc_round`（第幾次退回，1–3）。檢索仍只用原本的 `case_description`，退回理由附在 prompt 後面。
沒帶這兩個欄位時行為與先前完全相同。

## 檔案

| 檔案 | 內容 |
|---|---|
| `app/retrieval.py` | 01 v6 融合檢索、證據排版、輸出正規化（逐字沿用組員版本，請勿修改） |
| `app/gemini_agent.py` | `call_rules_agent()`、system prompt v5 |
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

Cloud Run → `tpl-rules-agent-gemini`（asia-southeast1）→「設定持續部署」→ 連結這個 repo、
分支 `main`、建構類型選 Dockerfile。之後 push 到 `main` 就會自動部署，服務原有的環境變數、
Secret、服務帳號、CPU／記憶體設定都會保留。
