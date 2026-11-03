# -*- coding: utf-8 -*-
"""TPL 規則代理人 v2：融合檢索 + 證據排版 + 輸出後處理。

本檔下半部的函式逐字取自組員交付的 01_TPL_規則代理人_融合檢索_單筆驗證_v6.ipynb
（cell 7/11/17/19/21/23/24/26/27），部署時只改兩處：
  1. 資料庫與模型改在 init() 載入一次（原本在 notebook 頂層執行）
  2. 拿掉 Gemini 呼叫與執行紀錄參數（交接手冊第 9 節：部署不得把案件內容送 Gemini）
format_evidence_for_prompt / _normalize_result 與 05 v6.1 第 4 節逐字相同（已比對），
任何一個字元不同都會造成訓練／推論不一致，請勿修改。
"""
import os, re, json, random, sqlite3
from typing import Any, Dict, List, Optional

import requests
import jieba
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = os.environ.get("TPL_RAG_DIR", "/app/artifacts/TPL_RAG_DB")
CHROMA_PERSIST_DIR = f"{DATA_DIR}/Vector_DB/chroma_db"
FINETUNED_MODEL_PATH = f"{DATA_DIR}/Vector_DB/e5_finetuned_tpl"
DRIVE_ROOT = f"{DATA_DIR}/BM25_DB"
TPL_CLAUSE_BM25_DB = f"{DRIVE_ROOT}/TPL_Clause_BM25_Tuned.db"
TPL_JUDGMENT_BM25_DB = f"{DRIVE_ROOT}/TPL_Judgment_BM25_Tuned.db"
TPL_CLAIM_BM25_DB = f"{DRIVE_ROOT}/TPL_Claim_BM25_Tuned.db"
BM25_DEFAULT_TOP_K = 5
BM25_TABLE_NAME = "bm25_ready"
BM25_ID_COLUMN = "global_chunk_id"
BM25_TEXT_COLUMN = "index_text"

BASE_URL = os.environ.get("GRAPHRAG_URL", "https://graphrag-api-624489584685.asia-east1.run.app")
API_KEY = os.environ.get("GRAPHRAG_API_KEY", "").strip()
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

model = policy_collection = case_collection = judgment_collection = None
GRAPH_STATUS = {"ok": None, "detail": "尚未檢查"}


def init(device: str = "cuda") -> dict:
    """啟動時呼叫一次：E5、ChromaDB、三個 BM25 庫（jieba 斷詞約 1.3 萬筆）、GraphRAG 健康檢查。"""
    global model, policy_collection, case_collection, judgment_collection
    model = SentenceTransformer(FINETUNED_MODEL_PATH, device=device)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    policy_collection = client.get_collection("tpl_policy")
    case_collection = client.get_collection("tpl_case")
    judgment_collection = client.get_collection("tpl_judgment")
    for name, path in (("clause", TPL_CLAUSE_BM25_DB), ("judgment", TPL_JUDGMENT_BM25_DB), ("claim", TPL_CLAIM_BM25_DB)):
        BM25_CORPORA[name] = _load_bm25_corpus(path)
    if not API_KEY:
        GRAPH_STATUS.update(ok=False, detail="未設定 GRAPHRAG_API_KEY，GraphRAG 來源會是空的")
    else:
        try:
            r = requests.get(f"{BASE_URL}/v1/health", headers=HEADERS, timeout=10)
            GRAPH_STATUS.update(ok=r.status_code == 200, detail=f"{r.status_code} {r.text[:200]}")
        except requests.RequestException as e:
            GRAPH_STATUS.update(ok=False, detail=f"{type(e).__name__}: {e}")
    return {"chroma": {"tpl_policy": policy_collection.count(), "tpl_case": case_collection.count(),
                       "tpl_judgment": judgment_collection.count()},
            "bm25": {k: len(v["texts"]) for k, v in BM25_CORPORA.items()}, "graphrag": dict(GRAPH_STATUS)}

# ---------- 01 v6 cell 7（逐字沿用）----------
# ========== 區塊四：查詢函式 ==========

def embed_query(text):
    """query 要加 'query:' 前綴，這是 e5 系列模型的慣例，別漏掉"""
    emb = model.encode([f"query: {text}"], normalize_embeddings=True)
    return emb[0].tolist()


def search_policy(query_text, top_k=5, exclude_article=("前言",)):
    """搜保單條款 / 強制險給付標準
    exclude_article：查詢端過濾掉封面/前言雜訊條目。用迴圈動態加大撈取範圍，
    不用固定倍數，因為前言污染筆數未知，固定倍數可能不夠用（例如撈9筆裡8筆都是前言）"""
    query_emb = embed_query(query_text)
    max_n = policy_collection.count()
    fetch_n = top_k

    while True:
        fetch_n = min(fetch_n, max_n)
        results = policy_collection.query(query_embeddings=[query_emb], n_results=fetch_n)
        filtered = [
            {"id": doc_id, "text": doc, "article": meta.get("article"), "source": meta.get("source"), "distance": dist}
            for doc_id, doc, meta, dist in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
            if meta.get("article") not in exclude_article
        ]
        if len(filtered) >= top_k or fetch_n >= max_n:
            return filtered[:top_k]
        fetch_n *= 3


def search_case(query_text, top_k=5):
    """搜理賠案件摘要（train+val合併）"""
    query_emb = embed_query(query_text)
    results = case_collection.query(query_embeddings=[query_emb], n_results=top_k)
    return [
        {"case_no": doc_id, "text": doc, "accident_area": meta.get("accident_area"),
         "own_fault_pct": meta.get("own_fault_pct"), "distance": dist}
        for doc_id, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def search_judgment_chunks(query_text, top_k=5):
    """搜判決書，回傳 chunk 層級結果（同一篇判決可能出現多筆）"""
    query_emb = embed_query(query_text)
    results = judgment_collection.query(query_embeddings=[query_emb], n_results=top_k)
    return [
        {"chunk_id": doc_id, "text": doc, "case_no": meta.get("case_no"), "distance": dist}
        for doc_id, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def search_judgment_by_case(query_text, top_k_chunks=20, top_k_docs=5):
    """搜判決書，聚合回「一篇判決一筆」（取每篇裡最相似的那個chunk代表整篇）"""
    from collections import defaultdict
    chunks = search_judgment_chunks(query_text, top_k=top_k_chunks)
    by_case = defaultdict(list)
    for c in chunks:
        by_case[c["case_no"]].append(c)

    aggregated = []
    for case_no, case_chunks in by_case.items():
        best = min(case_chunks, key=lambda x: x["distance"])
        aggregated.append({"case_no": case_no, "matched_chunk_id": best["chunk_id"],
                            "text": best["text"], "distance": best["distance"]})
    aggregated.sort(key=lambda x: x["distance"])
    return aggregated[:top_k_docs]

print("✅ 查詢函式定義完成：search_policy() / search_case() / search_judgment_chunks() / search_judgment_by_case()")


# ---------- 01 v6 cell 11（逐字沿用）----------
def _load_bm25_params(conn):
    """從 bm25_best_params 表讀該庫調好的 k1/b，讀不到就退回保守預設值"""
    try:
        cur = conn.cursor()
        cur.execute("SELECT k1, b FROM bm25_best_params LIMIT 1;")
        row = cur.fetchone()
        if row:
            return {"k1": row[0], "b": row[1]}
    except sqlite3.OperationalError:
        pass
    return {"k1": 1.5, "b": 0.75}


def _load_bm25_corpus(db_path):
    conn = sqlite3.connect(db_path)
    params = _load_bm25_params(conn)
    cur = conn.cursor()
    cur.execute(f"SELECT {BM25_ID_COLUMN}, {BM25_TEXT_COLUMN} FROM {BM25_TABLE_NAME};")
    rows = cur.fetchall()
    conn.close()

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    tokenized = [list(jieba.cut(t)) for t in texts]
    index = BM25Okapi(tokenized, k1=params["k1"], b=params["b"])
    return {"ids": ids, "texts": texts, "index": index, "params": params}


BM25_CORPORA = {}  # init() 載入


def bm25_sample_seed(n=5, corpus_name=None):
    """抽樣種子條款/判決/理賠段落。corpus_name=None 時三庫平均抽樣。
    回傳 [(text, global_chunk_id, corpus_name), ...]"""
    names = [corpus_name] if corpus_name else list(BM25_CORPORA.keys())
    per_corpus_n = max(1, n // len(names))
    seeds = []
    for name in names:
        corp = BM25_CORPORA[name]
        if not corp["texts"]:
            continue
        idxs = random.sample(range(len(corp["texts"])), min(per_corpus_n, len(corp["texts"])))
        for i in idxs:
            seeds.append((corp["texts"][i], corp["ids"][i], name))
    return seeds


def bm25_query(query, top_k=None, corpus_name=None):
    """跨庫（或指定單庫）檢索。回傳 [(text, score, global_chunk_id, corpus_name), ...]"""
    top_k = top_k or BM25_DEFAULT_TOP_K
    names = [corpus_name] if corpus_name else list(BM25_CORPORA.keys())
    tokenized_q = list(jieba.cut(query))
    all_hits = []
    for name in names:
        corp = BM25_CORPORA[name]
        if not corp["texts"]:
            continue
        scores = corp["index"].get_scores(tokenized_q)
        top_idx = scores.argsort()[::-1][:top_k]
        for i in top_idx:
            if scores[i] > 0:
                all_hits.append((corp["texts"][i], float(scores[i]), corp["ids"][i], name))
    all_hits.sort(key=lambda x: -x[1])
    return all_hits[:top_k]


# ---------- 01 v6 cell 17（逐字沿用）----------
# ---------------- GraphRAG ----------------
GRAPH_TIMEOUT = 30

# ---------------- 融合（496 題 QA 評估集調出的最佳配置）----------------
FUSION_METHOD = "weighted"          # "weighted" 或 "rrf"
FUSION_RRF_K = 60
FUSION_WEIGHT_BM25 = 0.5
FUSION_WEIGHT_VECTOR = 0.3
FUSION_WEIGHT_GRAPH = 0.2
FUSION_WEIGHTS = {"bm25": FUSION_WEIGHT_BM25, "vector": FUSION_WEIGHT_VECTOR, "graph": FUSION_WEIGHT_GRAPH}
FUSION_TOP_N_PER_CORPUS = 5
FUSION_TOP_N_GRAPH = 5

# ---------------- 輸出長度規範（與 01 v6 cell 17 相同）----------------
LLM_MAX_TRIGGERED_RULES = 3
LLM_MAX_EVIDENCE_CITED = 6
LLM_MAX_LIST_ITEM_CHARS = 150
LLM_MAX_REASONING_CHARS = 600




# ---------- 01 v6 cell 19（逐字沿用）----------
# ---------------- GraphRAG 呼叫函式 ----------------
def call_graphrag_case(case_no: str, top_k: int = 5, min_strength: int = 2,
                       min_jaccard: float = 0.15, timeout: int = GRAPH_TIMEOUT) -> Dict[str, Any]:
    """POST /v1/retrieve/case：已知案號 → case_info + 法院判決先例（precedents）"""
    clean_no = str(case_no).strip()
    if not clean_no.upper().startswith("TPL:"):
        clean_no = f"TPL:{clean_no}"
    payload = {"case_no": clean_no, "top_k": top_k, "min_strength": min_strength, "min_jaccard": min_jaccard}
    resp = requests.post(f"{BASE_URL}/v1/retrieve/case", headers=HEADERS, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def call_graphrag_tpl_similar(query_text: str, seed_case_no: str, top_k: int = 5,
                              timeout: int = GRAPH_TIMEOUT) -> Dict[str, Any]:
    """POST /v1/answer/tpl_similar：相似 TPL 理賠案件。
    ⚠️ API 限制：case_no 必須是真實存在、帶 TPL: 前綴的案號，不能給佔位符 → 沒有種子案號就不呼叫。"""
    clean_no = str(seed_case_no).strip()
    if not clean_no.upper().startswith("TPL:"):
        clean_no = f"TPL:{clean_no}"
    payload = {"case_no": clean_no, "question": query_text, "top_k": top_k}
    resp = requests.post(f"{BASE_URL}/v1/answer/tpl_similar", headers=HEADERS, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def call_graphrag_freetext(query_text: str, timeout: int = GRAPH_TIMEOUT) -> Dict[str, Any]:
    """POST /v1/answer/freetext：事故描述 → 特徵抽取 → 判決先例（非事故描述會回 400）"""
    payload = {"description": query_text, "question": "本案相關之法院判決先例為何？"}
    resp = requests.post(f"{BASE_URL}/v1/answer/freetext", headers=HEADERS, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def call_graphrag_clause(query_text: str, top_k: int = 5, timeout: int = GRAPH_TIMEOUT) -> Dict[str, Any]:
    """POST /v1/retrieve/clause：保單條款"""
    payload = {"query": query_text, "top_k": top_k}
    resp = requests.post(f"{BASE_URL}/v1/retrieve/clause", headers=HEADERS, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------- 圖節點格式化（全部用中括號／括號，不使用箭頭）----------------
def _format_case_node(case_info: Dict[str, Any]) -> str:
    injuries = case_info.get("injuries") or []
    bodyparts = case_info.get("bodyparts") or []
    regions = case_info.get("regions") or []
    own_fault_pct = case_info.get("own_fault_pct")
    attrs = [
        f"肇因={case_info.get('accident_cause') or '未知'}",
        f"肇責比例={own_fault_pct}%" if own_fault_pct is not None else "肇責比例=未知",
        f"肇責級距={case_info.get('fault_bucket') or '未知'}",
        f"傷勢={'、'.join(injuries) or '未知'}",
        f"部位={'、'.join(bodyparts) or '未知'}",
        f"地區={'、'.join(regions) or '未知'}",
    ]
    return f"[案件{case_info.get('case_no')}]（{'／'.join(attrs)}）"


def _format_precedent_node(p: Dict[str, Any]) -> str:
    verified_label = "已驗證主文" if p.get("verified") == 1 else "未驗證（法條推定）"
    attrs = [
        f"法院={p.get('court') or '未知'}",
        f"年度={p.get('year') or '未知'}",
        f"裁判類型={p.get('disposition_type') or '未知'}",
        f"強度={p.get('strength')}（{verified_label}）",
    ]
    if p.get("judgment_fault_pct") is not None:
        attrs.append(f"判決過失比例={p['judgment_fault_pct']}%（歸屬對象未經結構化確認，僅供參考）")
    return f"[{p.get('statute_no') or '未知案號'}]（{'／'.join(attrs)}）"


def _format_precedent_edge(p: Dict[str, Any]) -> str:
    jaccard = p.get("jaccard")
    confidence = "低信心參考" if (jaccard is not None and jaccard < 0.3) else "一般信心"
    matched = "、".join(p.get("matched_features") or []) or "無"
    return f"[相似度] jaccard={jaccard}（{confidence}），比對特徵=[{matched}]"


print("✅ GraphRAG 呼叫與格式化函式就緒")


# ---------- 01 v6 cell 21（逐字沿用）----------
def reciprocal_rank_fusion(ranked_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    fused_scores, doc_cache, doc_sources = {}, {}, {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            doc_id = item["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_sources.setdefault(doc_id, []).append(item["source"])
            doc_cache.setdefault(doc_id, item)
            if item.get("weak_relation"):
                doc_cache[doc_id]["weak_relation"] = True
    fused = [dict(doc_cache[d], fused_score=s, fused_from=doc_sources[d]) for d, s in fused_scores.items()]
    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused


def weighted_fusion(ranked_lists: List[List[Dict[str, Any]]], weights: List[float]) -> List[Dict[str, Any]]:
    fused_scores, doc_cache, doc_sources = {}, {}, {}
    for ranked, w in zip(ranked_lists, weights):
        if not ranked:
            continue
        max_score = max(item["score"] for item in ranked) or 1.0
        for item in ranked:
            doc_id = item["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + w * (item["score"] / max_score)
            doc_sources.setdefault(doc_id, []).append(item["source"])
            doc_cache.setdefault(doc_id, item)
            if item.get("weak_relation"):
                doc_cache[doc_id]["weak_relation"] = True
    fused = [dict(doc_cache[d], fused_score=s, fused_from=doc_sources[d]) for d, s in fused_scores.items()]
    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused


def fuse_corpus(corpus_name: str, source_lists: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """單一語料庫的 bm25 / vector / graph 三份排名融合成一份"""
    ranked_lists = list(source_lists.values())
    if FUSION_METHOD == "rrf":
        return reciprocal_rank_fusion(ranked_lists, k=FUSION_RRF_K)
    return weighted_fusion(ranked_lists, [FUSION_WEIGHTS.get(n, 0.0) for n in source_lists.keys()])


print(f"✅ 融合模組就緒：{FUSION_METHOD}，權重={FUSION_WEIGHTS}")


# ---------- 01 v6 cell 23（逐字沿用）----------
def _is_excluded(item: Dict[str, Any], exclude_case_nos: set) -> bool:
    """驗證時排除「案件本身」：驗證案件同時存在於 claim 語料庫（train+val 合併），
    若不排除，模型會直接看到自己的核定結果，驗證分數會虛高（資料洩漏）"""
    if not exclude_case_nos:
        return False
    blob = f"{item.get('id', '')} {item.get('text', '')}".upper()
    return any(cn.upper() in blob for cn in exclude_case_nos)


def graph_search_all(query_text: str, seed_case_no: Optional[str] = None, top_k: int = None,
                     exclude_case_nos: Optional[set] = None) -> Dict[str, List[Dict[str, Any]]]:
    """GraphDB 統一檢索轉接層（修正版）
    - claim：/v1/answer/tpl_similar（必須有真實種子案號才呼叫；similar_cases 空時改看 citations）
    - judgment：/v1/retrieve/case 的 precedents；precedents 為空（不是「dict 為空」）才退回 freetext
    - clause：/v1/retrieve/clause
    """
    top_k = top_k or FUSION_TOP_N_GRAPH
    exclude = set(exclude_case_nos or [])
    results = {"claim": [], "judgment": [], "clause": []}

    # ==================== 1. Claim ====================
    if seed_case_no:
        try:
            tpl_data = call_graphrag_tpl_similar(query_text, seed_case_no=seed_case_no, top_k=top_k + len(exclude))
            cases = tpl_data.get("similar_cases") or []
            if not cases:
                cases = [c for c in (tpl_data.get("citations") or []) if isinstance(c, dict) and c.get("case_no")]
            for idx, c in enumerate(cases, start=1):
                if not isinstance(c, dict):
                    continue
                c_no = c.get("case_no") or c.get("claim_id") or c.get("id") or f"GRAPH:CLAIM:{idx}"
                score = float(c.get("similarity") or c.get("score") or c.get("weighted_jaccard") or max(0.05, 1.0 - idx * 0.05))
                text_content = (_format_case_node(c) if ("accident_cause" in c or "injuries" in c)
                                else (c.get("content") or c.get("text") or json.dumps(c, ensure_ascii=False)))
                item = {"id": str(c_no), "text": f"[理賠案例] {text_content}", "score": score,
                        "source": "graph", "corpus": "claim"}
                if not _is_excluded(item, exclude):
                    results["claim"].append(item)
            results["claim"] = results["claim"][:top_k]
        except Exception as e:
            print(f"⚠️ GraphRAG tpl_similar 查詢失敗：{e}")

    # ==================== 2. Judgment ====================
    precedents = []
    if seed_case_no:
        try:
            precedents = call_graphrag_case(seed_case_no, top_k=top_k).get("precedents") or []
        except Exception as e:
            print(f"⚠️ GraphRAG /retrieve/case 查詢失敗：{e}")
    if not precedents and query_text:
        try:
            precedents = call_graphrag_freetext(query_text).get("precedents") or []
        except Exception as e:
            print(f"⚠️ GraphRAG freetext 查詢失敗（非事故描述會回 400，屬正常）：{e}")
    for idx, p in enumerate(precedents[:top_k], start=1):
        results["judgment"].append({
            "id": str(p.get("statute_no") or f"GRAPH:JUDGMENT:{idx}"),
            "text": f"[法院判決先例] {_format_precedent_node(p)} {_format_precedent_edge(p)}",
            "score": float(p.get("jaccard") or p.get("score") or 0.5),
            "source": "graph", "corpus": "judgment",
        })

    # ==================== 3. Clause ====================
    try:
        clause_data = call_graphrag_clause(query_text, top_k=top_k)
        clauses = clause_data.get("clauses") or clause_data.get("results") or []
        for idx, cl in enumerate(clauses[:top_k], start=1):
            title = cl.get("title") or cl.get("article") or ""
            content = cl.get("content") or cl.get("text") or str(cl)
            results["clause"].append({
                "id": str(cl.get("clause_id") or cl.get("id") or f"GRAPH:CLAUSE:{idx}"),
                "text": f"[保單條款] {title} {content}".strip(),
                "score": float(cl.get("score") or cl.get("similarity") or 0.8),
                "source": "graph", "corpus": "clause",
            })
    except Exception as e:
        print(f"⚠️ GraphRAG clause 查詢失敗：{e}")

    return results


# ---------- 01 v6 cell 24（逐字沿用）----------
# 標準 12 碼理賠案號（例如 0111V2500749）
CASE_NO_REGEX = re.compile(r"\b\d{4}[A-Za-z]\d{7}\b")


def _extract_real_case_no(items: List[Dict[str, Any]]) -> Optional[str]:
    """沒有傳入 seed_case_no 時（例如純文字查詢），才從 claim 命中項目的 ID / 內文萃取案號"""
    for it in items:
        for field in ("id", "text"):
            m = CASE_NO_REGEX.search(str(it.get(field, "")))
            if m:
                return m.group(0).upper()
    return None


# ---------------- 檢索轉接層：BM25 / Vector 輸出 → 融合共用格式 ----------------
def bm25_search_corpus(corpus_name: str, query: str, top_k: int = None) -> List[Dict[str, Any]]:
    top_k = top_k or FUSION_TOP_N_PER_CORPUS
    hits = bm25_query(query, top_k=top_k, corpus_name=corpus_name)
    return [{"id": h[2], "score": float(h[1]), "text": h[0], "source": "bm25", "corpus": corpus_name} for h in hits]


def vector_search_corpus(corpus_name: str, query: str, top_k: int = None) -> List[Dict[str, Any]]:
    top_k = top_k or FUSION_TOP_N_PER_CORPUS
    if corpus_name == "clause":
        rs = search_policy(query, top_k=top_k)
        return [{"id": str(r["id"]), "score": 1.0 / (1.0 + float(r["distance"])), "text": r["text"],
                 "source": "vector", "corpus": "clause"} for r in rs]
    if corpus_name == "claim":
        rs = search_case(query, top_k=top_k)
        return [{"id": str(r["case_no"]), "score": 1.0 / (1.0 + float(r["distance"])), "text": r["text"],
                 "source": "vector", "corpus": "claim"} for r in rs]
    if corpus_name == "judgment":
        rs = search_judgment_by_case(query, top_k_docs=top_k)
        return [{"id": str(r["case_no"]), "score": 1.0 / (1.0 + float(r["distance"])), "text": r["text"],
                 "source": "vector", "corpus": "judgment"} for r in rs]
    return []


def _print_source_rankings(label: str, by_corpus: Dict[str, List[Dict[str, Any]]]):
    print(f"\n[{label} 各自排名]")
    for corp_name, items in by_corpus.items():
        print(f"  [{corp_name}]")
        if not items:
            print("    (無結果)")
        for rank, it in enumerate(items, start=1):
            print(f"    [{label}-{rank}] [分數={it['score']:.4f}] [ID={it['id']}] {it['text'][:45]}...")


# ==================== 整合檢索流程 ====================
def fusion_retrieve(query: str,
                    entity_keywords: Optional[List[str]] = None,
                    seed_case_no: Optional[str] = None,
                    exclude_case_nos: Optional[List[str]] = None,
                    verbose: bool = True) -> Dict[str, Any]:
    """三來源（BM25 / Vector / GraphDB）× 三語料庫融合檢索
    - seed_case_no：驗證已知歷史案件時直接傳真實案號，不再從檢索結果用正規表示式猜
    - exclude_case_nos：驗證時排除案件本身（避免資料洩漏）
    """
    top_n = FUSION_TOP_N_PER_CORPUS
    exclude = set(exclude_case_nos or [])
    fetch_n = top_n + (3 if exclude else 0)      # 多撈幾筆，排除自己後仍保有 top_n
    evidence: Dict[str, Any] = {"query": query, "corpora": {}}

    # 1. BM25 / Vector
    bm25_by_corpus, vector_by_corpus = {}, {}
    for c in ("claim", "clause", "judgment"):
        bm25_by_corpus[c] = [x for x in bm25_search_corpus(c, query, top_k=fetch_n) if not _is_excluded(x, exclude)][:top_n]
        vector_by_corpus[c] = [x for x in vector_search_corpus(c, query, top_k=fetch_n) if not _is_excluded(x, exclude)][:top_n]

    # 2. 種子案號：有傳就用；沒傳才從 claim 命中結果萃取
    if not seed_case_no:
        seed_case_no = _extract_real_case_no(bm25_by_corpus["claim"] + vector_by_corpus["claim"])
    evidence["graph_seed_case_no"] = seed_case_no

    # 3. GraphDB
    graph_by_corpus = graph_search_all(query, seed_case_no=seed_case_no, top_k=FUSION_TOP_N_GRAPH,
                                       exclude_case_nos=exclude)

    if verbose:
        _print_source_rankings("BM25", bm25_by_corpus)
        _print_source_rankings("Vector", vector_by_corpus)
        _print_source_rankings("Graph", graph_by_corpus)

    # 4. 語料庫內融合（BM25 + Vector + GraphDB）
    for corpus_name in ("claim", "clause", "judgment"):
        source_lists = {"bm25": bm25_by_corpus[corpus_name],
                        "vector": vector_by_corpus[corpus_name],
                        "graph": graph_by_corpus[corpus_name]}
        evidence["corpora"][corpus_name] = {
            "bm25_raw": source_lists["bm25"],
            "vector_raw": source_lists["vector"],
            "graph_raw": source_lists["graph"],
            "fused_top": fuse_corpus(corpus_name, source_lists)[:top_n],
        }
    return evidence


CORPUS_LABELS = {"claim": "相似理賠案例", "clause": "保單條款", "judgment": "法院判決"}


def print_fused_evidence(evidence: Dict[str, Any], text_chars: int = 100):
    """印出各語料庫融合後排名（中括號格式，與餵給 LLM 的編號一致）"""
    print(f"\n[圖資料庫種子案號] {evidence.get('graph_seed_case_no')}")
    for corpus_name, label in CORPUS_LABELS.items():
        items = evidence["corpora"].get(corpus_name, {}).get("fused_top", [])
        print(f"\n[{label}]（BM25+Vector+GraphDB 融合前 {len(items)} 筆）")
        if not items:
            print("  (無結果)")
        for i, it in enumerate(items, 1):
            weak = " [弱關聯]" if it.get("weak_relation") else ""
            print(f"[{corpus_name}-{i}] [融合分={it['fused_score']:.4f}] "
                  f"[來源={'+'.join(sorted(set(it['fused_from'])))}]{weak} {it['text'][:text_chars]}...")


print("✅ fusion_retrieve() 就緒（支援 seed_case_no / exclude_case_nos）")


# ---------- 01 v6 cell 26（逐字沿用）----------
# ==============================================================================
# format_evidence_for_prompt：全系統唯一的證據格式化函式（中括號版）
# ------------------------------------------------------------------------------
# - 每筆證據一律用 [clause-1] / [claim-2] / [judgment-3] 中括號標籤，不再使用 ▶ 或 → 箭頭
# - call_rules_agent()、單筆驗證、之後任何要重建訓練資料的地方都必須呼叫這一份，
#   逐字元一致，避免 Train-Test Mismatch
# - ⚠️ 若 Notebook ⑤/⑥ 或 v2_1 adapter 訓練時用的中括號排版與這裡有任何字元差異，
#   請以訓練時的那一份為準，直接整段貼過來取代這個函式
# ==============================================================================
EVIDENCE_CORPUS_ORDER = [
    ("clause", "相關保單條款"),
    ("claim", "相似理賠案例"),
    ("judgment", "法院判決先例"),
]


def _get_fused_items(evidence: Any, corpus_key: str) -> List[Dict[str, Any]]:
    corpora_data = evidence.get("corpora", evidence) if isinstance(evidence, dict) else {}
    corp_info = corpora_data.get(corpus_key, {})
    if isinstance(corp_info, dict):
        return corp_info.get("fused_top") or corp_info.get("fused") or []
    if isinstance(corp_info, list):
        return corp_info
    return []


def format_evidence_for_prompt(evidence: Any) -> str:
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except Exception:
            return "（無法解析檢索證據）"
    if not isinstance(evidence, dict):
        return "（無檢索證據）"

    blocks = []
    for corpus_key, title in EVIDENCE_CORPUS_ORDER:
        items = _get_fused_items(evidence, corpus_key)
        lines = [f"【{title}】"]
        if not items:
            lines.append(f"[{corpus_key}] (無資料)")
        for idx, it in enumerate(items, start=1):
            score = it.get("fused_score") or it.get("score") or 0.0
            src_list = it.get("fused_from") or [it.get("source", "fusion")]
            if isinstance(src_list, str):
                src_list = [src_list]
            src_str = "+".join(sorted(set(str(s) for s in src_list)))
            raw_text = it.get("text") or it.get("content") or str(it)
            clean_text = " ".join(raw_text.replace("\r", "\n").split())   # 壓成單行，避免雜亂換行
            lines.append(f"[{corpus_key}-{idx}] (來源: {src_str} | 融合分: {score:.4f})：{clean_text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks).strip()


def valid_evidence_ids(evidence: Dict[str, Any]) -> set:
    """本次 prompt 裡實際存在的證據編號，用來過濾模型亂編的 evidence_cited"""
    ids = set()
    for corpus_key, _ in EVIDENCE_CORPUS_ORDER:
        for idx in range(1, len(_get_fused_items(evidence, corpus_key)) + 1):
            ids.add(f"{corpus_key}-{idx}")
    return ids


print("✅ format_evidence_for_prompt（中括號版，全系統唯一）已就緒")


# ---------- 01 v6 cell 27（逐字沿用）----------
# ==============================================================================
# LLM 輸出後處理：正規化 / 截斷救援 / 系統錯誤標記
# ==============================================================================
DECISION_LABELS = ["拒賠", "理賠", "資料不足", "疑似詐欺"]
_LIST_FIELDS = ["triggered_rules", "missing_data", "fraud_indicators", "evidence_cited"]
_EVID_ID_RE = re.compile(r"(clause|claim|judgment)\s*-\s*(\d+)", re.I)


def _cap_str(s: Any, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _dedupe(seq):
    out, seen = [], set()
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _normalize_result(res: Dict[str, Any], valid_ids: Optional[set] = None) -> Dict[str, Any]:
    """把 LLM 輸出整理成固定 schema，並把長度鎖在安全範圍內（防止下游再被撐爆）"""
    out = dict(res)
    if out.get("decision") not in DECISION_LABELS:
        out["decision"] = "資料不足"
        out["_decision_invalid"] = True
    try:
        out["confidence"] = max(0.0, min(1.0, float(out.get("confidence", 0.0))))
    except (TypeError, ValueError):
        out["confidence"] = 0.0

    for key in _LIST_FIELDS:
        v = out.get(key) or []
        if isinstance(v, str):
            v = [x for x in re.split(r"[；;\n]", v) if x.strip()]
        out[key] = [str(x).strip() for x in v if str(x).strip()]

    # triggered_rules：一條一個規則、限制條數與字數
    out["triggered_rules"] = _dedupe(_cap_str(x, LLM_MAX_LIST_ITEM_CHARS) for x in out["triggered_rules"])[:LLM_MAX_TRIGGERED_RULES]
    out["fraud_indicators"] = _dedupe(_cap_str(x, LLM_MAX_LIST_ITEM_CHARS) for x in out["fraud_indicators"])[:LLM_MAX_TRIGGERED_RULES]

    # evidence_cited：只留編號本體（clause-1），去掉中括號，並剔除 prompt 裡不存在的編號
    cited = []
    for x in out["evidence_cited"]:
        m = _EVID_ID_RE.search(x)
        if m:
            cid = f"{m.group(1).lower()}-{int(m.group(2))}"
            if valid_ids is None or cid in valid_ids:
                cited.append(cid)
    out["evidence_cited"] = _dedupe(cited)[:LLM_MAX_EVIDENCE_CITED]

    # missing_data：必須是具體文件名稱，且不可跟 evidence_cited 重疊
    out["missing_data"] = [
        _cap_str(x, LLM_MAX_LIST_ITEM_CHARS) for x in _dedupe(out["missing_data"])
        if not _EVID_ID_RE.fullmatch(x.strip("[] "))
    ][:LLM_MAX_TRIGGERED_RULES]

    out["reasoning"] = _cap_str(out.get("reasoning") or "", LLM_MAX_REASONING_CHARS)
    return out


def _salvage_truncated_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """JSON 被截斷時，把「已經完整寫出來」的欄位救回來。
    只收完整的字串（有結尾引號），寫到一半的那一條直接丟掉，不猜內容。"""
    if not raw_text:
        return None
    m = re.search(r'"decision"\s*:\s*"(' + "|".join(DECISION_LABELS) + r')"', raw_text)
    if not m:
        return None
    res: Dict[str, Any] = {"decision": m.group(1)}
    m = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', raw_text)
    res["confidence"] = float(m.group(1)) if m else 0.0

    def _unescape(s):
        try:
            return json.loads(f'"{s}"')
        except Exception:
            return s

    m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_text, re.S)
    res["reasoning"] = _unescape(m.group(1)) if m else ""
    for key in _LIST_FIELDS:
        seg = re.search(rf'"{key}"\s*:\s*\[(.*?)(?:\]|$)', raw_text, re.S)
        res[key] = [_unescape(s) for s in re.findall(r'"((?:[^"\\]|\\.)*)"', seg.group(1))] if seg else []
    return res


def _fallback_result(raw_text: str, reason: str) -> Dict[str, Any]:
    """完全救不回來時的保底結果。
    注意：missing_data 保持空陣列，不再塞「LLM 輸出格式錯誤」這種占位文字，
    錯誤資訊只放在底線開頭欄位 + is_system_error，避免污染評估與訓練資料。"""
    return {
        "decision": "資料不足",
        "confidence": 0.0,
        "triggered_rules": [],
        "reasoning": "",
        "missing_data": [],
        "fraud_indicators": [],
        "evidence_cited": [],
        "is_system_error": True,
        "_error_reason": reason,
        "_raw_error": (raw_text or "")[:1000],
    }
