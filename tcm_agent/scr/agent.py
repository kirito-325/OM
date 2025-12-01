"""中医诊断信息提取智能体模块"""

import json
import sys
import os

# 添加父目录到路径，以便导入同级模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm import call_llm
from prompt import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT,
    VALIDATION_SYSTEM_PROMPT,
    VALIDATION_USER_PROMPT,
    DIAGNOSIS_SYSTEM_PROMPT,
    DIAGNOSIS_USER_PROMPT
)


def tcm_sydrom_agent(case_dict: dict, max_retries: int = 3) -> dict:
    """
    中医诊断信息提取智能体

    该函数接收包含多种中医诊断信息的字典作为输入，提取所需的症状信息，
    最终返回一个结构化的字典。

    Args:
        case_dict: 包含中医诊断信息的字典，例如：
            {
                "tcm_check": "神清，精神可，心态平和，语声中等，气息平和，舌淡红苔薄白，脉弦细。",
                "tcm_evidence": "患者素体禀赋不足..."
            }
        max_retries: 最大重试次数，默认为3

    Returns:
        结构化的症状信息字典，格式如下：
        {
            "inspection": {
                "mental_state": ["神清", "精神可", "心态平和"],
                "voice": ["语声中等"],
                "breath": ["气息平和"],
                "tongue": {
                    "tongue_body": "淡红",
                    "tongue_coating": "薄白"
                }
            },
            "palpation": {
                "pulse": ["弦", "细"]
            },
            "subjective_symptoms": ["关节疼痛", "肿胀", ...],
            "oral_findings": ["龋齿"]
        }
    """
    # 1. 遍历输入字典，整合成一条文本
    combined_text = ""
    for _, value in case_dict.items():
        if isinstance(value, str):
            combined_text += f"{value}\n"
        else:
            combined_text += f"{json.dumps(value, ensure_ascii=False)}\n"
    combined_text = combined_text.strip()

    # 2. 构建提取消息
    extraction_messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": EXTRACTION_USER_PROMPT.format(text=combined_text)}
    ]

    extracted_result = {}

    # 3. 循环提取和验证
    for attempt in range(max_retries):
        print(f"第 {attempt + 1} 次提取尝试...")

        # 调用提取LLM
        extracted_result = call_llm(extraction_messages)

        if not extracted_result:
            print("提取失败，重试中...")
            continue

        # 构建验证消息
        validation_messages = [
            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": VALIDATION_USER_PROMPT.format(
                original_text=combined_text,
                extracted_result=json.dumps(extracted_result, ensure_ascii=False, indent=2)
            )}
        ]

        # 调用验证LLM
        print("正在验证提取结果...")
        validation_result = call_llm(validation_messages)

        # 检查验证结果
        if validation_result.get("is_valid", False):
            print("验证通过！")
            return extracted_result

        # 验证失败，输出问题
        print(f"验证未通过:")
        if validation_result.get("missing_items"):
            print(f"  - 遗漏项: {validation_result['missing_items']}")
        if validation_result.get("wrong_items"):
            print(f"  - 错误项: {validation_result['wrong_items']}")
        if validation_result.get("suggestions"):
            print(f"  - 建议: {validation_result['suggestions']}")

        # 如果还有重试机会，将问题反馈给提取LLM重新处理
        if attempt < max_retries - 1:
            feedback_parts = []
            if validation_result.get("missing_items"):
                feedback_parts.append(f"遗漏了以下信息：{validation_result['missing_items']}")
            if validation_result.get("wrong_items"):
                feedback_parts.append(f"以下信息有误：{validation_result['wrong_items']}")
            if validation_result.get("suggestions"):
                feedback_parts.append(f"修改建议：{validation_result['suggestions']}")

            feedback = "；".join(feedback_parts) if feedback_parts else "请重新仔细提取"

            extraction_messages.append({
                "role": "assistant",
                "content": json.dumps(extracted_result, ensure_ascii=False)
            })
            extraction_messages.append({
                "role": "user",
                "content": f"上次提取结果存在问题。{feedback}。请根据原始文本重新提取，确保不遗漏任何症状信息。"
            })

    # 超过最大重试次数，返回最后一次的结果
    print(f"已达到最大重试次数({max_retries})，返回当前结果")
    return extracted_result

def tcm_diagnosis_agent(case_dict: dict) -> dict:
    """
    中医诊断智能体，根据症状信息推测病名和证型

    Args:
        case_dict: 结构化的症状信息字典（来自 tcm_sydrom_agent 的输出）

    Returns:
        诊断结果，格式：{"think": "推理过程", "tcm_diagnosis": "病名-证型"}
    """
    # 1. 将症状字典转换为文本描述
    symptoms_text = _format_symptoms(case_dict)

    # 2. 构建诊断消息
    diagnosis_messages = [
        {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
        {"role": "user", "content": DIAGNOSIS_USER_PROMPT.format(symptoms=symptoms_text)}
    ]

    # 3. 直接调用LLM进行诊断
    print("正在进行病证诊断...")
    diagnosis_result = call_llm(diagnosis_messages)

    if diagnosis_result and "tcm_diagnosis" in diagnosis_result:
        print(f"诊断完成：{diagnosis_result['tcm_diagnosis']}")
        if "think" in diagnosis_result:
            print(f"推理过程：{diagnosis_result['think']}")
    else:
        print("诊断失败，返回空结果")
        diagnosis_result = {"think": "", "tcm_diagnosis": ""}

    return diagnosis_result

def tcm_treatment_agent(case_dict: dict, tcm_diagnosis: dict, max_retries: int = 2) -> dict:
    """中医给方智能体（ReAct 风格）

    采用 ReAct 循环：LLM 每轮返回 {"thought", "action", "action_input"}，
    agent 执行 action（调用对应 prompt），把 observation 反馈回 LLM，直到 action 为 finish。
    """
    # 组织病例文本用于 prompt
    symptoms_text = _format_symptoms(case_dict)

    steps = []

    def _call_with_retry(messages):
        for attempt in range(max_retries):
            res = call_llm(messages)
            if res:
                return res
        return {}

    # 从 prompt 导入 ReAct 所需提示词
    from prompt import (
        TREATMENT_REACT_SYSTEM_PROMPT,
        TREATMENT_DETERMINE_PRINCIPLE_PROMPT,
        TREATMENT_SELECT_BASE_PROMPT,
        TREATMENT_PROPOSE_MODIFICATIONS_PROMPT,
        TREATMENT_DETERMINE_DOSAGE_PROMPT,
    )

    # 初始对话
    react_messages = [
        {"role": "system", "content": TREATMENT_REACT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"tcm_diagnosis": tcm_diagnosis, "symptoms": symptoms_text}, ensure_ascii=False)}
    ]

    final_prescription = {
        "tcm_diagnosis": tcm_diagnosis,
        "tcm_treatment_principle": "",
        "base_formula": {},
        "modifications": [],
        "dosage": [],
        "useway": ""
    }

    max_steps = 8
    for step_idx in range(max_steps):
        print(f"ReAct 第 {step_idx+1} 步：向 LLM 请求下一步动作...")
        react_res = _call_with_retry(react_messages)
        if not isinstance(react_res, dict):
            print("LLM 未返回有效 JSON，终止 ReAct 流程")
            break

        action = react_res.get("action")
        thought = react_res.get("thought", "")
        action_input = react_res.get("action_input", {}) or {}

        steps.append({"step": f"react_step_{step_idx+1}", "think": thought, "action": action, "action_input": action_input})

        observation = {}

        if action == "determine_principle":
            messages = [
                {"role": "system", "content": TREATMENT_DETERMINE_PRINCIPLE_PROMPT},
                {"role": "user", "content": TREATMENT_DETERMINE_PRINCIPLE_PROMPT.format(tcm_diagnosis=json.dumps(tcm_diagnosis, ensure_ascii=False), symptoms=symptoms_text)}
            ]
            observation = _call_with_retry(messages) or {}
            final_prescription["tcm_treatment_principle"] = observation.get("tcm_treatment_principle", "")

        elif action == "select_base_formula":
            context = {"tcm_diagnosis": tcm_diagnosis, "treatment_principle": final_prescription.get("tcm_treatment_principle", ""), "symptoms": symptoms_text}
            messages = [
                {"role": "system", "content": TREATMENT_SELECT_BASE_PROMPT},
                {"role": "user", "content": TREATMENT_SELECT_BASE_PROMPT.format(context=json.dumps(context, ensure_ascii=False))}
            ]
            observation = _call_with_retry(messages) or {}
            final_prescription["base_formula"] = observation.get("base_formula", {})

        elif action == "propose_modifications":
            context = {"symptoms": symptoms_text, "tcm_diagnosis": tcm_diagnosis, "base_formula": final_prescription.get("base_formula", {})}
            messages = [
                {"role": "system", "content": TREATMENT_PROPOSE_MODIFICATIONS_PROMPT},
                {"role": "user", "content": TREATMENT_PROPOSE_MODIFICATIONS_PROMPT.format(context=json.dumps(context, ensure_ascii=False))}
            ]
            observation = _call_with_retry(messages) or {}
            final_prescription["modifications"] = observation.get("modifications", [])

        elif action == "determine_dosage":
            final_herbs = []
            base_herbs = final_prescription.get("base_formula", {}).get("herbs", []) or []
            final_herbs.extend(base_herbs)
            for m in final_prescription.get("modifications", []):
                if isinstance(m, dict):
                    herb_name = m.get("herb")
                    if herb_name and herb_name not in final_herbs:
                        final_herbs.append(herb_name)

            context = {"herbs": final_herbs}
            messages = [
                {"role": "system", "content": TREATMENT_DETERMINE_DOSAGE_PROMPT},
                {"role": "user", "content": TREATMENT_DETERMINE_DOSAGE_PROMPT.format(context=json.dumps(context, ensure_ascii=False))}
            ]
            observation = _call_with_retry(messages) or {}
            final_prescription["dosage"] = observation.get("dosage", [])
            final_prescription["useway"] = observation.get("useway", "")

        elif action == "finish":
            final_summary = action_input or {}
            for k in ("tcm_treatment_principle", "base_formula", "modifications", "dosage", "useway"):
                if k in final_summary and final_summary[k]:
                    final_prescription[k] = final_summary[k]
            steps.append({"step": "finished", "think": thought, "action_input": action_input})
            break

        else:
            print(f"未知动作: {action}, 终止。")
            break

        react_messages.append({"role": "assistant", "content": json.dumps(observation, ensure_ascii=False)})
        react_messages.append({"role": "user", "content": '收到观测结果，请继续下一步（只返回 JSON: {"thought":..., "action":..., "action_input":{...}}）。'})

    result = {"steps": steps, "final_prescription": final_prescription}
    return result

def _format_symptoms(case_dict: dict) -> str:
    """将症状字典格式化为易读的文本"""
    parts = []

    # 望诊
    inspection = case_dict.get("inspection", {})
    if inspection:
        if inspection.get("mental_state"):
            parts.append(f"神志：{'、'.join(inspection['mental_state'])}")
        if inspection.get("voice"):
            parts.append(f"语声：{'、'.join(inspection['voice'])}")
        if inspection.get("breath"):
            parts.append(f"气息：{'、'.join(inspection['breath'])}")
        tongue = inspection.get("tongue", {})
        if tongue.get("tongue_body") or tongue.get("tongue_coating"):
            tongue_desc = f"舌{tongue.get('tongue_body', '')}苔{tongue.get('tongue_coating', '')}"
            parts.append(tongue_desc)

    # 切诊
    palpation = case_dict.get("palpation", {})
    if palpation.get("pulse"):
        parts.append(f"脉{''.join(palpation['pulse'])}")

    # 主观症状
    symptoms = case_dict.get("subjective_symptoms", [])
    if symptoms:
        parts.append(f"症状：{'、'.join(symptoms)}")

    # 口腔
    oral = case_dict.get("oral_findings", [])
    if oral:
        parts.append(f"口腔：{'、'.join(oral)}")

    return "；".join(parts) if parts else json.dumps(case_dict, ensure_ascii=False)
