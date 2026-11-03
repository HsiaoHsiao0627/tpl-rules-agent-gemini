# -*- coding: utf-8 -*-
"""TPL 規則代理人 v2 API（Cloud Run L4）。

POST /v1/tpl/rules  {"case_description": "<build_case_query 格式的案件描述>"}
     → 融合檢索 fusion_retrieve() → 本地 adapter rules_agent_local() → 正規化 JSON
     每件約 1 分鐘（檢索 5–15 秒＋生成 40–90 秒）；一張 GPU 一次處理一件，其餘排隊。
GET  /health        模型與檢索資源是否載入完成（可用來預熱；不要用 /healthz，Cloud Run 會在前端攔掉結尾是 z 的路徑）

欄位 case_description 與舊版 tpl-rules-agent-api 相同，舊的 claim-intake 也能直接呼叫。
2026-11-03：選填 judge_feedback／recalc_round——法官代理人不同意時，claim-intake-v2 帶著退回理由重打本端點
（最多退回 3 次）。檢索仍只用原本的 case_description（證據不因退回意見而改變），退回意見附在 prompt 的
案件描述之後，請模型依同樣證據重新檢查；沒帶 judge_feedback 時行為與先前完全相同。
回應比舊版多：is_system_error / _error_reason（前端必須顯示為「系統錯誤，需人工複核」）、
evidence（本次 prompt 裡每個證據編號對應的內容，讓前端能顯示 [clause-1] 是什麼）、elapsed。
"""
import os, time, threading
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

API_KEY = os.environ.get("API_KEY", "").strip()
app = FastAPI(title="TPL 規則代理人 v2（Gemma 4 12B adapter）")
_state = {"ready": False, "error": None, "info": {}}
_lock = threading.Lock()


class RulesRequest(BaseModel):
    case_description: str
    seed_case_no: Optional[str] = None   # 已知的歷史案號（驗證用）；新案件留空
    judge_feedback: Optional[str] = None # 法官代理人的退回理由（退回重算時才帶）
    recalc_round: Optional[int] = None   # 第幾次退回（1–3）


def _with_judge_feedback(case_description: str, feedback: Optional[str], rnd: Optional[int]) -> str:
    """退回重算：把法官的退回理由附在案件描述後面；證據與判斷順序不變，不得因此臆測證據沒寫的事實。"""
    if not (feedback or "").strip():
        return case_description
    head = f"[法官代理人退回意見]（第 {rnd} 次退回重新判斷）" if rnd else "[法官代理人退回意見]"
    return (f"{case_description}\n\n{head}\n"
            "法官代理人審核你前一次的判斷與後續理賠結果後，提出下列問題。請依同樣的證據與判斷順序重新檢查本案："
            "問題確實成立就修正結論；不成立就維持原結論，並在 reasoning 說明理由。"
            "不得因為這些意見而臆測證據沒有寫到的事實。\n"
            f"{feedback.strip()}")


def _load():
    try:
        t0 = time.time()
        from . import retrieval
        _state["info"]["retrieval"] = retrieval.init(device="cpu")
        from . import gemini_agent
        _state["info"]["model"] = {"llm": gemini_agent.LLM_MODEL_NAME, "prompt": "v5"}
        _state["info"]["load_sec"] = round(time.time() - t0)
        _state["ready"] = True
        print(f"[啟動] 完成 {_state['info']}", flush=True)
    except Exception as e:  # 讓 /health 看得到原因
        _state["error"] = f"{type(e).__name__}: {e}"
        print(f"[啟動] 失敗 {_state['error']}", flush=True)


@app.on_event("startup")
def _startup():
    threading.Thread(target=_load, daemon=True).start()


@app.get("/health")
@app.get("/healthz")   # 本機用；Cloud Run 上這個路徑到不了容器
def healthz():
    return {"ready": _state["ready"], "error": _state["error"], "info": _state["info"]}


def _evidence_summary(evidence: dict) -> dict:
    from .retrieval import EVIDENCE_CORPUS_ORDER, _get_fused_items
    out = {}
    for key, _title in EVIDENCE_CORPUS_ORDER:
        out[key] = [{"label": f"{key}-{i}", "id": str(it.get("id")), "sources": sorted(set(it.get("fused_from") or [])),
                     "score": round(float(it.get("fused_score") or 0), 4), "text": " ".join(str(it.get("text") or "").split())[:400]}
                    for i, it in enumerate(_get_fused_items(evidence, key), start=1)]
    return out


@app.post("/v1/tpl/rules")
def rules(req: RulesRequest, x_api_key: Optional[str] = Header(None)):
    if API_KEY and (x_api_key or "").strip() != API_KEY:
        raise HTTPException(401, "API key 錯誤")
    if _state["error"]:
        raise HTTPException(500, f"模型載入失敗：{_state['error']}")
    if not _state["ready"]:
        raise HTTPException(503, "模型載入中，請稍後再試（冷啟動約 5–10 分鐘）")
    if not req.case_description.strip():
        raise HTTPException(400, "case_description 不可為空")
    from .retrieval import fusion_retrieve
    from .gemini_agent import call_rules_agent
    t0 = time.time()
    if True:  # Gemini 版不佔 GPU，可並行
        evidence = fusion_retrieve(req.case_description, seed_case_no=req.seed_case_no, verbose=False)
        t1 = time.time()
        result = call_rules_agent(_with_judge_feedback(req.case_description, req.judge_feedback, req.recalc_round), evidence)
        result.pop("_meta", None)
    result["evidence"] = _evidence_summary(evidence)
    result["graph_seed_case_no"] = evidence.get("graph_seed_case_no")
    result["needs_manual_review"] = bool(result.get("is_system_error")) or result.get("decision") in ("拒賠", "疑似詐欺")
    result["elapsed"] = {"retrieval_sec": round(t1 - t0, 1), "total_sec": round(time.time() - t0, 1)}
    result["recalc_round"] = req.recalc_round if (req.judge_feedback or "").strip() else None
    result["version"] = "tpl-rules-agent-gemini（01 v6 融合檢索 + Gemini，prompt v5；支援法官退回意見）"
    print(f"[案件] round={result['recalc_round']} decision={result.get('decision')} system_error={result.get('is_system_error')} "
          f"reason={result.get('_error_reason')} {result['elapsed']}", flush=True)
    return result
