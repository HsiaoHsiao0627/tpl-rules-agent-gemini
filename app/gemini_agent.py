# -*- coding: utf-8 -*-
"""TPL 規則代理人（Gemini 版）：01 v8 第 8 節 call_rules_agent()。

system prompt 由環境變數 PROMPT_VERSION 決定：
  v8（預設）：01 v8 notebook 驗證用的 prompt，逐字沿用（含 Isolation Forest 標記與 [詐欺偵測規則] 的解讀方式）
  v5        ：2026-10-30 上線的四步驟 prompt（回退用）
"""
import os, re, json, time
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from .retrieval import (format_evidence_for_prompt, valid_evidence_ids, _normalize_result,
                        _salvage_truncated_json, _fallback_result, DECISION_LABELS)
from .fraud import FRAUD_FLAG_ENABLED, FRAUD_RULE_ENABLED, FRAUD_FLAG_TEXT, FRAUD_RULE_TEXT

# ---------------- Gemini 規則代理人 ----------------
LLM_MODEL_NAME = "gemini-3.6-flash"
LLM_API_KEY_ENV = "GEMINI_API_KEY"
LLM_TEMPERATURE = 0.1
LLM_MAX_OUTPUT_TOKENS = 8192        # 一次給足；截斷時不是再拉高，而是改用精簡模式重寫
LLM_THINKING_LEVEL = "low"          # thinking token 也算在 max_output_tokens 內；None = 不設定
LLM_USE_RESPONSE_SCHEMA = True      # 用 schema 的 max_items 從生成端鎖住陣列長度
LLM_MAX_TRIGGERED_RULES = 3
LLM_MAX_EVIDENCE_CITED = 6
LLM_MAX_LIST_ITEM_CHARS = 150
LLM_MAX_REASONING_CHARS = 600


RULES_AGENT_SYSTEM_PROMPT_V5 = '你是台灣第三人責任保險理賠案件的「規則代理人」(Rules Agent)。\n你的職責是根據提供的證據（相似理賠案例、保單條款、法院判決），判斷本案的「責任與承保」是否成立，\n並在下列四種狀態中擇一。\n\n請「依序」檢查，前一步成立就停止，不再往下：\n\n第一步\u3000疑似詐欺：只要出現下列任一「跡象」就判疑似詐欺（不需要確證，最終由人工複核）：\n  - 短期內多次理賠、同一傷勢或同一車損重複求償、與過去理賠案件高度重複\n  - 被保險人與受害人往來異常密切，或雙方相約、串通共同向保險公司爭取理賠\n  - 指定或可疑的修車廠、醫療院所，或與特定業者往來異常\n  - 請求金額或治療明顯偏離傷勢程度（例如扭傷、挫傷等輕傷卻長期住院、多次手術、長期看護或高額費用；修復費接近車價）\n  - 文件日期、內容互相矛盾（例如診斷日期早於事故日、說法前後不一）\n  - 保戶與受害人關係特殊、事故經過不合常理、保單生效後 30 日內出險\n  - 證據（含圖資料庫、相似案例）顯示異常關聯\n  fraud_indicators 必須逐條列出看到的跡象。\n\n第二步\u3000拒賠：符合下列任一情形，判拒賠，並在 triggered_rules 寫出對應的不保事項：\n  - 本車無肇事責任（肇責 0%、判定無肇事責任、或證據顯示完全由對方造成）：被保險人依法不負賠償責任\n  - 駕駛人飲酒後駕駛，吐氣或血液酒精濃度超過道路交通管理法規標準（共同條款第九條第九款）\n  - 駕駛人吸毒、服用違禁藥物後駕駛（共同條款第九條第七款）\n  - 無照駕駛、駕照吊扣吊銷期間駕駛、越級駕駛等違反道路交通管理處罰條例第二十一條、第二十一條之一、\n    第二十二條第一項第一款至第六款或第二項之情形（共同條款第九條第六款）\n  - 被保險人或駕駛人故意或唆使之行為、駕車從事犯罪或逃避逮捕（共同條款第九條第四款、第八款）\n  - 被保險汽車出租或作有償載客載貨使用所致（共同條款第九條第五款）\n  - 受害人不是第三人：乘坐或上下被保險汽車之人、被保險人或駕駛人本人、其家長、家屬、\n    執行職務中之受僱人（第三人責任保險第五條第二款、第三款）\n  - 請求的不是第三人責任險承保的賠償責任（例如被保險汽車本身的車體損失、保費、減費或延長保期、\n    保險公司對他人的代位求償）：這類請求判拒賠，reasoning 說明不屬本險承保範圍\n  - 同一案件同時有「第三人的損失」與「不屬本險的請求」（例如本車車損）時，以第三人部分判斷，\n    不屬本險的部分在 reasoning 註明不賠即可，不因此整案拒賠\n\n第三步\u3000資料不足：只有在缺少「判斷責任與承保所必需」的資料時才使用，限於：\n  缺事故證明且無肇責判定、肇責尚待鑑定、缺診斷證明書且無傷勢描述、傷勢類型為失能卻無失能等級、\n  傷勢類型為死亡卻無死亡證明\n\n第四步\u3000理賠：前三步都不成立，且符合下列全部條件：\n  (a) 有事故證明或肇責判定，且本車有肇事責任（肇責比例大於 0%）\n  (b) 受害人有診斷證明書（或傷勢描述）可確認傷害與事故相關\n  (c) 沒有觸發第一步、第二步的任何情形\n\n判斷原則（必須遵守）：\n- 金額相關文件不屬於本代理人的判斷範圍：不得因為缺少醫療收據、費用明細、請求金額、薪資扣繳憑單、\n  和解書、調解書、判決書、估價單而判「資料不足」。這些文件由理賠代理人與理賠人員在核算金額時處理。\n- 只能就「本案實際有請求的項目」考慮文件：沒有請求工作損失，不得提到薪資證明；\n  精神慰撫金不需要精神科診斷或心理評估文件。\n- 案件描述中沒有「事故經過」文字，不構成資料不足；以事故證明或肇責判定為準。\n- 判「理賠」時，若有後續核算金額需要的文件，可在 reasoning 末尾以「後續核算需備：…」簡述，但 missing_data 必須為空陣列。\n- 證據編號（如 [clause-2]、[judgment-1]）只能用來支持「該證據內容實際寫到的事」；\n  上面這些判斷原則不是證據，說明時不得替它們加上證據編號。\n\n請務必只依據提供的證據作判斷，不可臆測未提供的事實。\n你的輸出只做規則觸發判斷與嫌疑標記，最終理賠核定由其他代理人負責，因此無需給出金額或最終賠付結論。\n\n輸出長度規範（必須遵守）：\n- triggered_rules 最多 3 條，只列「真正決定本案判斷」的條款或規則，\n  一個陣列元素只放一條，每條 150 字以內；\n  禁止逐條列舉保單所有條文，禁止把多條條文用分號串在同一個元素裡。\n- reasoning 250 字以內，引用證據時使用中括號編號，例如 [clause-1]、[judgment-2]。\n- evidence_cited 最多 6 個，只放編號本身（例如 clause-1），不放內文。\n- missing_data 只在資料不足時列出具體缺少的文件名稱（例如「肇事責任鑑定書」），否則為空陣列。\n\n請以下列 JSON 格式輸出（不要有其他文字、不要用 markdown code fence）：\n{\n  "decision": "拒賠 | 理賠 | 資料不足 | 疑似詐欺",\n  "confidence": 0.0 到 1.0 之間的浮點數,\n  "triggered_rules": ["觸發的具體條款或規則描述"],\n  "reasoning": "簡要說明判斷依據，需引用證據編號（如 [clause-1]、[judgment-2]）",\n  "missing_data": ["若資料不足，列出缺漏項目；否則為空陣列"],\n  "fraud_indicators": ["若疑似詐欺，列出具體指標；否則為空陣列"],\n  "evidence_cited": ["實際引用的證據編號列表"]\n}\n'   # prompt v5（2026-10-30，Gemini 驗證版）


# ---------------- prompt v8（01 v8 第 8 節，逐字沿用；開關關閉時對應段落為空字串）----------------
_FRAUD_FLAG_RULE = (
    f"\n關於「{FRAUD_FLAG_TEXT}」：相似理賠案例若帶有這個標記，代表該歷史案件被異常偵測模組（Isolation Forest）列為離群，"
    "需人工審核，並非確認詐欺。這只是參考資訊，"
    + ("本案是否疑似詐欺一律以案件描述中的「[詐欺偵測規則]」為準。\n" if FRAUD_RULE_ENABLED else
       "若本案在受害人數、理賠事故結次、就醫紀錄、險種結次、保險金額上也明顯偏離一般案件，可判定為疑似詐欺；不可只因檢索到標記案例就判定。\n")
) if FRAUD_FLAG_ENABLED else ""

_FRAUD_RULE_RULE = (
    f"\n關於案件描述中的「{FRAUD_RULE_TEXT}」：這是異常偵測模組以 696 件歷史訓練案件學出的規則，只分「離群」與「主群」兩類。\n"
    "- 判定為「離群」時：decision 必須為「疑似詐欺」；fraud_indicators 逐條寫出符合的規則代號、條件與本案數值"
    "（例如「規則 A：理賠事故結次 4 筆 > 2.5 且險種結次 4 筆 <= 5」）；reasoning 說明本案需轉人工審核，並非確認詐欺。\n"
    "- 判定為「主群」時：decision 不可為「疑似詐欺」（即使檢索到帶標記的案例），請依保單條款與證據判斷「理賠」、「拒賠」或「資料不足」，"
    "fraud_indicators 為空陣列。\n"
) if FRAUD_RULE_ENABLED else ""

RULES_AGENT_SYSTEM_PROMPT_V8 = f"""你是台灣第三人責任保險理賠案件的「規則代理人」(Rules Agent)。
你的職責是根據提供的證據（相似理賠案例、保單條款、法院判決），
判斷本案應觸發下列四種狀態之一：

1. "拒賠" - 明確違反保單條款或除外責任，有充分證據支持拒賠
2. "理賠" - 符合理賠條件，未觸發任何拒賠規則，資料充分
3. "資料不足" - 現有證據無法充分判斷，需補件或補充調查
4. "疑似詐欺" - 存在詐欺指標（如圖資料庫顯示異常關聯、重複理賠模式、證據矛盾）
{_FRAUD_FLAG_RULE}{_FRAUD_RULE_RULE}
請務必只依據提供的證據作判斷，不可臆測未提供的事實。
你的輸出只做規則觸發判斷與嫌疑標記，最終理賠核定由其他代理人負責，因此無需給出金額或最終賠付結論。

輸出長度規範（必須遵守）：
- triggered_rules 最多 {LLM_MAX_TRIGGERED_RULES} 條，只列「真正決定本案判斷」的條款或規則，
  一個陣列元素只放一條，每條 {LLM_MAX_LIST_ITEM_CHARS} 字以內；
  禁止逐條列舉保單所有條文，禁止把多條條文用分號串在同一個元素裡。
- reasoning 250 字以內，引用證據時使用中括號編號，例如 [clause-1]、[judgment-2]。
- evidence_cited 最多 {LLM_MAX_EVIDENCE_CITED} 個，只放編號本身（例如 clause-1），不放內文。
- missing_data 只在資料不足時列出具體缺少的文件名稱（例如「肇事責任鑑定書」），否則為空陣列。

請以下列 JSON 格式輸出（不要有其他文字、不要用 markdown code fence）：
{{
  "decision": "拒賠 | 理賠 | 資料不足 | 疑似詐欺",
  "confidence": 0.0 到 1.0 之間的浮點數,
  "triggered_rules": ["觸發的具體條款或規則描述"],
  "reasoning": "簡要說明判斷依據，需引用證據編號（如 [clause-1]、[judgment-2]）",
  "missing_data": ["若資料不足，列出缺漏項目；否則為空陣列"],
  "fraud_indicators": ["若疑似詐欺，列出具體指標；否則為空陣列"],
  "evidence_cited": ["實際引用的證據編號列表"]
}}
"""

PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v8").strip().lower()
RULES_AGENT_SYSTEM_PROMPT = RULES_AGENT_SYSTEM_PROMPT_V5 if PROMPT_VERSION == "v5" else RULES_AGENT_SYSTEM_PROMPT_V8
PROMPT_VERSION = "v5" if PROMPT_VERSION == "v5" else "v8"

# 截斷後的重試提示：不是單純把 token 上限拉高（模型進入「逐條列舉條文」迴圈時，給再多 token 也寫不完），
# 而是明確要求精簡後重寫
COMPACT_RETRY_SUFFIX = """

## 輸出長度限制（重要）
上一次的輸出過長而被截斷。請重新輸出完整 JSON，並嚴格遵守：
- triggered_rules 最多 2 條，每條 40 字以內，只列決定本案判斷的條款，禁止列舉所有條文
- reasoning 150 字以內
- evidence_cited 最多 5 個編號
"""

# ---- 結構化輸出 schema：用 max_items 從生成端直接鎖住陣列長度 ----
RULES_AGENT_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "decision": types.Schema(type=types.Type.STRING, enum=DECISION_LABELS),
        "confidence": types.Schema(type=types.Type.NUMBER),
        "triggered_rules": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING),
                                        max_items=LLM_MAX_TRIGGERED_RULES),
        "reasoning": types.Schema(type=types.Type.STRING),
        "missing_data": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING),
                                     max_items=LLM_MAX_TRIGGERED_RULES),
        "fraud_indicators": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING),
                                         max_items=LLM_MAX_TRIGGERED_RULES),
        "evidence_cited": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING),
                                       max_items=LLM_MAX_EVIDENCE_CITED),
    },
    required=["decision", "confidence", "triggered_rules", "reasoning",
              "missing_data", "fraud_indicators", "evidence_cited"],
    property_ordering=["decision", "confidence", "triggered_rules", "reasoning",
                       "missing_data", "fraud_indicators", "evidence_cited"],
)

_gemini_key = os.environ.get(LLM_API_KEY_ENV, "").strip()
if not _gemini_key:
    raise RuntimeError("未設定 GEMINI_API_KEY")
gemini_client = genai.Client(api_key=_gemini_key)

# 執行期旗標：第一次呼叫若因 schema / thinking 設定被 API 拒絕，自動關閉並提示，不讓整批中斷
_LLM_RUNTIME = {"use_schema": LLM_USE_RESPONSE_SCHEMA, "thinking_level": LLM_THINKING_LEVEL}


def _build_gen_config(max_output_tokens: int) -> types.GenerateContentConfig:
    kwargs = dict(
        system_instruction=RULES_AGENT_SYSTEM_PROMPT,
        temperature=LLM_TEMPERATURE,
        response_mime_type="application/json",
        max_output_tokens=max_output_tokens,
    )
    if _LLM_RUNTIME["use_schema"]:
        kwargs["response_schema"] = RULES_AGENT_RESPONSE_SCHEMA
    if _LLM_RUNTIME["thinking_level"]:
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=_LLM_RUNTIME["thinking_level"])
        except Exception:
            # SDK 版本太舊不認得 thinking_level → 不設定
            _LLM_RUNTIME["thinking_level"] = None
    return types.GenerateContentConfig(**kwargs)


def _finish_reason_name(response) -> Optional[str]:
    try:
        fr = response.candidates[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return None
    return getattr(fr, "name", None) or (str(fr) if fr is not None else None)


def _usage_dict(response) -> Dict[str, Any]:
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return {}
    return {
        "prompt_tokens": getattr(um, "prompt_token_count", None),
        "output_tokens": getattr(um, "candidates_token_count", None),
        "thinking_tokens": getattr(um, "thoughts_token_count", None),   # thinking 也會吃 max_output_tokens
    }


def _call_gemini_once(user_prompt: str, max_output_tokens: int):
    try:
        response = gemini_client.models.generate_content(
            model=LLM_MODEL_NAME, contents=user_prompt, config=_build_gen_config(max_output_tokens),
        )
    except Exception as e:
        msg = str(e)
        # 400 且跟 schema / thinking 有關 → 關掉進階設定再試一次
        if ("400" in msg or "INVALID_ARGUMENT" in msg) and (_LLM_RUNTIME["use_schema"] or _LLM_RUNTIME["thinking_level"]):
            print(f"⚠️ API 不接受 response_schema / thinking 設定，改用基本設定重試：{msg[:150]}")
            _LLM_RUNTIME["use_schema"] = False
            _LLM_RUNTIME["thinking_level"] = None
            response = gemini_client.models.generate_content(
                model=LLM_MODEL_NAME, contents=user_prompt, config=_build_gen_config(max_output_tokens),
            )
        else:
            raise
    raw_text = ""
    try:
        raw_text = (response.text or "").strip()
    except Exception:
        raw_text = ""
    raw_text = re.sub(r"^```(?:json)?|```$", "", raw_text).strip()
    return raw_text, _finish_reason_name(response), _usage_dict(response)


def _try_parse(raw_text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(raw_text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def call_rules_agent(query: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    規則代理人統一呼叫介面。截斷處理流程：
      第 1 次：正常 prompt（max_output_tokens=LLM_MAX_OUTPUT_TOKENS、schema 鎖陣列長度、thinking 壓低）
      第 2 次：若被截斷(MAX_TOKENS)或 JSON 無效 → 加上「精簡模式」指示重寫（同樣上限，不是只拉高上限）
      第 3 步：仍失敗 → 從殘缺 JSON 救回已完整的欄位（標記 _salvaged + is_system_error）
      最後：完全救不回 → _fallback_result（is_system_error=True，不污染業務欄位）
    任何結果都會經過 _normalize_result() 把長度鎖住，並附上 _meta（嘗試次數、finish_reason、token 用量）。
    """
    evidence_text = format_evidence_for_prompt(evidence)
    base_prompt = f"## 案件描述\n{query}\n\n## 檢索證據\n{evidence_text}"
    valid_ids = valid_evidence_ids(evidence)
    attempts_meta = []

    prompts = [base_prompt, base_prompt + COMPACT_RETRY_SUFFIX]
    raw_text, finish_reason = "", None
    for attempt, prompt in enumerate(prompts, start=1):
        raw_text, finish_reason, usage = _call_gemini_once(prompt, LLM_MAX_OUTPUT_TOKENS)
        attempts_meta.append({"attempt": attempt, "finish_reason": finish_reason, **usage})
        parsed = _try_parse(raw_text)
        if parsed is not None and finish_reason != "MAX_TOKENS":
            result = _normalize_result(parsed, valid_ids)
            result["is_system_error"] = False
            result["_meta"] = {"attempts": attempts_meta}
            return result
        why = "被截斷(MAX_TOKENS)" if finish_reason == "MAX_TOKENS" else "JSON 無法解析"
        if attempt < len(prompts):
            print(f"⚠️ 第 {attempt} 次輸出{why}，改用精簡模式重試...")

    # 兩次都失敗 → 嘗試救回殘缺 JSON
    reason = "truncated_output" if finish_reason == "MAX_TOKENS" else "invalid_json"
    salvaged = _salvage_truncated_json(raw_text)
    if salvaged is not None:
        result = _normalize_result(salvaged, valid_ids)
        result.update({"is_system_error": True, "_salvaged": True, "_error_reason": f"{reason}_salvaged",
                       "_raw_error": raw_text[:1000]})
    else:
        result = _fallback_result(raw_text, reason)
    result["_meta"] = {"attempts": attempts_meta}
    print(f"❌ 兩次皆失敗（{reason}），{'已救回部分欄位' if salvaged else '使用保底結果'}，已標記 is_system_error，需人工複核")
    return result


print(f"✅ 規則代理人 LLM 模組就緒（prompt {PROMPT_VERSION}；schema 限長 + thinking 壓低 + 精簡重試 + 截斷救援 + 輸出正規化）")
