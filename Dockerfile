# TPL 規則代理人（Gemini 版）— GitHub 來源部署用
#
# 以線上現行映像為底：Python 套件版本（含 chromadb，必須與 chroma_db 建立時的版本一致）、
# E5 模型、ChromaDB、三個 BM25 資料庫都沿用底層映像 /app/artifacts，這裡只覆蓋程式碼 app/。
# 底層映像由 9_TPL_RulesAgent_Gemini_Deploy 區塊四建出；若要換資料或升級套件，
# 先用那份 notebook 重建底層映像，再把下面的標籤改成新的。
FROM asia-southeast1-docker.pkg.dev/cki-scu-2026/tpl-rules/api:gemini-v5

ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ANONYMIZED_TELEMETRY=False \
    TPL_RAG_DIR=/app/artifacts/TPL_RAG_DB
WORKDIR /app
RUN rm -rf /app/app
COPY app/ /app/app/
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
