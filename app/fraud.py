# -*- coding: utf-8 -*-
"""v8 詐欺偵測：Isolation Forest 離群標記 + 主群／離群兩大類規則（01 v8 第 7、10b 節）。

1. annotate_fraud_flags()：檢索到 train 中被 Isolation Forest 標為離群的 36 件時，內文前面加上
   [異常偵測標記：疑似詐欺]（只標 claim 語料庫）。案號清單讀 app/data/train_if_outliers.txt
   （由 tpl_if_outlier_labels.csv 中 split=train 且 離群標記=1 的案號產生，一行一個）。
   val 的標記是驗證答案，絕對不能放進這個檔案。
2. build_fraud_rule_lines()：依 4 個紀錄筆數欄位算規則 A／B，產生附在案件描述最後的
   [詐欺偵測規則] 區塊（只給 Gemini，不進檢索）。規則與
   TPL_IF_Outlier_Rules_Validation_兩大類.xlsx「規則條件_兩大類」完全相同：
     離群 A：理賠事故結次筆數 > 2.5 且 險種結次筆數 <= 5
     離群 B：理賠事故結次筆數 > 1.5 且 險種結次筆數 > 5 且（就醫紀錄筆數 > 3.5 或 傷害險保額合計 > 25,000,000）
     其餘  ：主群
"""
import os, re
from typing import Any, Dict, List, Optional

FRAUD_FLAG_ENABLED = os.environ.get("FRAUD_FLAG_ENABLED", "1") != "0"
FRAUD_RULE_ENABLED = os.environ.get("FRAUD_RULE_ENABLED", "1") != "0"
FRAUD_FLAG_TEXT = "[異常偵測標記：疑似詐欺]"
FRAUD_RULE_TEXT = "[詐欺偵測規則]"

CASE_NO_REGEX = re.compile(r"\b\d{4}[A-Za-z]\d{7}\b")
_OUTLIER_FILE = os.environ.get(
    "TRAIN_OUTLIER_FILE", os.path.join(os.path.dirname(__file__), "data", "train_if_outliers.txt"))


def _load_train_outliers(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8-sig") as f:
        return {ln.strip().upper() for ln in f if CASE_NO_REGEX.fullmatch(ln.strip())}


TRAIN_FRAUD_CASE_NOS = _load_train_outliers(_OUTLIER_FILE)
if FRAUD_FLAG_ENABLED and not TRAIN_FRAUD_CASE_NOS:
    print(f"⚠️ FRAUD_FLAG_ENABLED=1 但找不到 train 離群案號清單（{_OUTLIER_FILE}），檢索結果不會加標記", flush=True)


def _item_case_no(item: Dict[str, Any]) -> Optional[str]:
    for field in ("id", "text"):
        m = CASE_NO_REGEX.search(str(item.get(field, "")))
        if m:
            return m.group(0).upper()
    return None


def annotate_fraud_flags(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """只標 train 的已知離群案件（claim 語料庫的 fused_top）。"""
    n = 0
    if FRAUD_FLAG_ENABLED and TRAIN_FRAUD_CASE_NOS:
        for it in evidence["corpora"]["claim"]["fused_top"]:
            cn = _item_case_no(it)
            if cn and cn in TRAIN_FRAUD_CASE_NOS and not str(it.get("text", "")).startswith(FRAUD_FLAG_TEXT):
                it["text"] = f"{FRAUD_FLAG_TEXT} {it.get('text', '')}"
                it["fraud_flag"] = True
                n += 1
    evidence["n_fraud_flagged_claims"] = n
    return evidence


# ---------------- 規則 A／B ----------------
FRAUD_RULE_THRESHOLDS = {
    "A_episode_gt": 2.5, "A_seq_le": 5,
    "B_episode_gt": 1.5, "B_seq_gt": 5, "B_visit_gt": 3.5, "B_injury_gt": 25_000_000,
}
FRAUD_RULES_TEXT = {
    "A": "理賠事故結次筆數 > 2.5 且 險種結次筆數 <= 5",
    "B": "理賠事故結次筆數 > 1.5 且 險種結次筆數 > 5 且（就醫紀錄筆數 > 3.5 或 傷害險保額合計 > 25,000,000 元）",
}


def rule_features(rc: Dict[str, Any]) -> Dict[str, Any]:
    rc = rc or {}
    return {
        "episode_count": int(rc.get("episode_count") or 0),       # claim_episodes 筆數
        "seq_count": int(rc.get("seq_count") or 0),               # coverages 筆數（不去重）
        "visit_count": int(rc.get("visit_count") or 0),           # medical_visits 筆數
        "injury_insured": float(rc.get("injury_insured") or 0),   # coverages 31 / 33 險種保額加總
    }


def fraud_rule_check(record_counts: Dict[str, Any]) -> Dict[str, Any]:
    f, t = rule_features(record_counts), FRAUD_RULE_THRESHOLDS
    hit_a = f["episode_count"] > t["A_episode_gt"] and f["seq_count"] <= t["A_seq_le"]
    hit_b = (f["episode_count"] > t["B_episode_gt"] and f["seq_count"] > t["B_seq_gt"]
             and (f["visit_count"] > t["B_visit_gt"] or f["injury_insured"] > t["B_injury_gt"]))
    group = "A" if hit_a else ("B" if hit_b else None)          # A、B 互斥（險種結次 <= 5 與 > 5）
    return {"result": "離群" if group else "主群", "group": group, "hit_a": hit_a, "hit_b": hit_b, "features": f}


def build_fraud_rule_lines(record_counts: Dict[str, Any]) -> List[str]:
    """與 01 v8 build_case_query() 最後附加的區塊逐字相同。"""
    if not FRAUD_RULE_ENABLED:
        return []
    r = fraud_rule_check(record_counts)
    f = r["features"]
    yn = lambda b: "符合" if b else "不符合"
    return [
        "",
        f"{FRAUD_RULE_TEXT}（異常偵測模組以 696 件歷史訓練案件學出的規則，只分「離群」與「主群」）",
        f"[判定] {r['result']}" + (f"（符合規則 {r['group']}）" if r["group"] else "（規則 A、B 皆不符合）"),
        f"[規則 A] {FRAUD_RULES_TEXT['A']}｜本案 理賠事故結次 {f['episode_count']} 筆、險種結次 {f['seq_count']} 筆｜{yn(r['hit_a'])}",
        f"[規則 B] {FRAUD_RULES_TEXT['B']}｜本案 理賠事故結次 {f['episode_count']} 筆、險種結次 {f['seq_count']} 筆、"
        f"就醫紀錄 {f['visit_count']} 筆、傷害險保額合計 {f['injury_insured']:,.0f} 元｜{yn(r['hit_b'])}",
    ]


def split_rule_block(case_description: str) -> (str, Optional[str]):
    """呼叫端若已在案件描述最後附好 [詐欺偵測規則] 區塊，拆成（檢索用描述, 規則區塊）。"""
    idx = case_description.find(FRAUD_RULE_TEXT)
    if idx < 0:
        return case_description, None
    return case_description[:idx].rstrip(), case_description[idx:].strip()
