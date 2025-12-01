"""
中医诊断信息提取流水线

该脚本调用 scr/agent.py 中的 diagnosis_agent 函数，
传入包含中医诊断信息的字典，提取结构化的症状信息。

输出结构示例：
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
  "subjective_symptoms": [
    "关节疼痛",
    "肿胀",
    "形体消瘦",
    "口干",
    "欲饮",
    "眼干",
    "鼻干"
  ],
  "oral_findings": [
    "龋齿"
  ]
}
"""

import sys
import os
import json

# 添加 scr 目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scr'))

from agent import tcm_sydrom_agent
from agent import tcm_diagnosis_agent

def main():
    # 测试用例
    case = {"tcm_check": "得神，心态平和，语声清晰，气息畅，舌红苔黄，脉细数。",
      "tcm_evidence": "患者素体禀赋不足，年老体弱，饮食失调日久，易致气阴两虚，气阴两虚则肌肤筋骨关节失于濡养，病邪留恋，闭阻经脉，深伏关节故关节疼痛、肿胀。气虚失运，生化乏源，则形体消瘦，口眼干燥，口干欲饮等症。阴虚则津不上承，故口齿失养，出现龋齿。舌质红少苔，脉沉细，均为气阴两虚之象。"
      }

    print("=" * 50)
    print("开始提取中医诊断信息...")
    print("=" * 50)

    # 1. 调用症状提取智能体
    symptoms = tcm_sydrom_agent(case)
    print("\n症状提取结果：")
    print(json.dumps(symptoms, ensure_ascii=False, indent=2))

    # 2. 调用诊断智能体（基于提取的症状）
    print("\n" + "=" * 50)
    print("开始进行病证诊断...")
    print("=" * 50)
    diagnosis = tcm_diagnosis_agent(symptoms)

    print("\n" + "=" * 50)
    print("最终诊断结果：")
    print("=" * 50)
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))

    # 合并输出
    output = {**symptoms, **diagnosis}
    return output


if __name__ == "__main__":
    main()
