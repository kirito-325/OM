"""中医诊断信息提取提示词模块"""

# ==================== 症状提取提示词 ====================

EXTRACTION_SYSTEM_PROMPT = """你是一个专业的中医信息提取助手。你的任务是从中医诊断文本中准确提取结构化的症状信息。

要求：
1. 只提取文本中明确提及的症状，不要推断或添加
2. 严格按照指定的JSON格式输出
3. 如果某个字段在文本中没有相关信息，使用空列表[]或空字符串""
4. 不要输出任何解释性文字，只返回JSON对象"""


EXTRACTION_USER_PROMPT = """请从以下中医诊断信息中提取症状信息：

{text}

请按照以下分类提取信息：

1. **inspection（望诊）**：
   - mental_state: 神志、精神状态（如：神清、精神可、心态平和、神疲、萎靡等）
   - voice: 语声相关（如：语声中等、语声低微、声音洪亮等）
   - breath: 气息相关（如：气息平和、气短、气促等）
   - tongue: 舌象
     - tongue_body: 舌质（如：淡红、红、暗红、淡白、紫暗等）
     - tongue_coating: 舌苔（如：薄白、黄腻、少苔、白腻、厚腻等）

2. **palpation（切诊）**：
   - pulse: 脉象，提取所有脉象描述词（如：弦、细、沉、数、滑、涩、弱、无力等）

3. **subjective_symptoms（主观症状）**：
   - 患者的主观感受和症状（如：疼痛、肿胀、口干、眼干、乏力、失眠、消瘦等）

4. **oral_findings（口腔情况）**：
   - 口腔相关发现（如：龋齿、牙龈出血、口腔溃疡等）

请严格按照以下JSON格式输出：
{{
  "inspection": {{
    "mental_state": [],
    "voice": [],
    "breath": [],
    "tongue": {{
      "tongue_body": "",
      "tongue_coating": ""
    }}
  }},
  "palpation": {{
    "pulse": []
  }},
  "subjective_symptoms": [],
  "oral_findings": []
}}

注意：
- 脉象如"脉弦细"应拆分为["弦", "细"]
- 舌象如"舌淡红苔薄白"应解析为tongue_body="淡红", tongue_coating="薄白"
- 只返回JSON对象，不要有其他文字"""


# ==================== 验证提示词 ====================

VALIDATION_SYSTEM_PROMPT = """你是一个中医信息验证助手。你的任务是验证从中医诊断文本中提取的症状信息是否准确和完整。

验证要点：
1. 提取的信息是否确实来自原始文本
2. 是否有遗漏的重要症状信息
3. 分类是否正确
4. 舌脉信息解析是否正确"""


VALIDATION_USER_PROMPT = """请验证以下提取结果是否正确：

【原始文本】
{original_text}

【提取结果】
{extracted_result}

请检查：
1. 提取的每一项信息是否都能在原始文本中找到依据
2. 原始文本中的症状信息是否被完整提取
3. 各项信息的分类是否正确（望诊、切诊、主观症状、口腔情况）
4. 舌象和脉象的解析是否准确

请以JSON格式返回验证结果：
{{
  "is_valid": true或false,
  "missing_items": ["遗漏的信息1", "遗漏的信息2"],
  "wrong_items": ["错误的信息1", "错误的信息2"],
  "suggestions": "具体的修改建议"
}}

如果提取结果完全正确，则is_valid为true，其他字段为空列表或空字符串。
只返回JSON对象。"""


# ==================== 病证诊断提示词 ====================

DIAGNOSIS_SYSTEM_PROMPT = """你是资深中医诊断专家，精通辨证论治。根据患者四诊信息（望闻问切），综合分析后给出病名和证型诊断。

要求：
1. 先在 think 字段中详细分析辨证思路
2. 然后给出最终诊断结果"""


DIAGNOSIS_USER_PROMPT = """【四诊信息】
{symptoms}

请按以下步骤分析并输出JSON：

1. **think**: 详细写出你的辨证分析过程，包括：
   - 主症分析：患者的主要症状是什么
   - 舌脉分析：舌象和脉象提示什么
   - 病机推断：综合分析病因病机
   - 辨证依据：为什么得出这个病-证诊断

2. **tcm_diagnosis**: 最终诊断，格式为"病名-证型"

【输出格式】
{{
  "think": "辨证分析过程...",
  "tcm_diagnosis": "病名-证型"
}}

【示例】
{{
  "think": "患者主症为关节疼痛、肿胀，伴口眼干燥、口干欲饮。舌红少苔，脉细数。关节疼痛肿胀属痹证范畴，结合口眼干燥等津液不足表现，考虑燥痹。舌红少苔、脉细数为阴虚内热之象，故辨证为阴虚内热证。",
  "tcm_diagnosis": "燥痹-阴虚内热证"
}}

只返回JSON对象。"""


# ==================== 给方（Treatment）相关提示词（ReAct 风格） ====================

TREATMENT_REACT_SYSTEM_PROMPT = """你是资深中医临床处方专家。你的交互采用 ReAct 模式：每次返回一个 JSON 对象，包含三个字段：
1) "thought": 你的短期推理（中文），
2) "action": 要执行的动作名（见下面的可用动作），
3) "action_input": 动作所需输入（JSON 对象）。

可用动作：
- determine_principle: 根据辨病与症状确定治则/治法，返回 {"tcm_treatment_principle":"..."}
- select_base_formula: 推荐一个基础方，返回 {"base_formula": {"name":"","source":"","herbs":[...]}}
- propose_modifications: 针对基础方提出加减，返回 {"modifications": [{"herb":"...","reason":"..."}, ...]}
- determine_dosage: 为最终药物给出用量与煎服，返回 {"dosage": [{"herb":"...","dose":"...g"}], "useway":"..."}
- finish: 表示流程结束，action_input 可包含最终处方摘要

每次只执行一个动作，动作由环境（agent）执行并把结果作为 "observation" 反馈给你，随后你继续下一步思考并选择下一个动作或 `finish`。

返回示例：
{
  "thought": "根据患者阴虚内热，首要滋阴清热，同时通络止痛",
  "action": "determine_principle",
  "action_input": {}
}

只返回严格的 JSON 对象，不要额外的说明文本。"""


TREATMENT_DETERMINE_PRINCIPLE_PROMPT = """根据以下信息确定中医治则/治法并以 JSON 返回：
辨病信息：{tcm_diagnosis}
病人主要症状：{symptoms}

返回格式：{"tcm_treatment_principle": "...", "think": "简短推理"}
只返回 JSON 对象。"""


TREATMENT_SELECT_BASE_PROMPT = """根据辨病/治则以及病人症状推荐一个基础方（name, source, herbs list），并说明选择理由。\n
输入：{context}

返回格式：{"base_formula": {"name":"","source":"","herbs":[...]}, "think":"..."}
只返回 JSON 对象。"""


TREATMENT_PROPOSE_MODIFICATIONS_PROMPT = """根据下列病人症状与基础方，提出需要增减的草药并说明理由。
输入：{context}

返回格式：{"modifications":[{"herb":"...","reason":"..."}, ...], "think":"..."}
只返回 JSON 对象。"""


TREATMENT_DETERMINE_DOSAGE_PROMPT = """为给定的药物列表提供建议用量（以 g 为单位）及煎服方法。
输入：{context}

返回格式：{"dosage":[{"herb":"...","dose":"...g"}, ...], "useway":"煎服方法说明", "think":"..."}
只返回 JSON 对象。"""
