import os
from openai import OpenAI


def get_response(messages):
    client = OpenAI(
        api_key="sk-114601cde4d14dadb8008a7c20e7c591",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    completion = client.chat.completions.create(model="qwen-plus", messages=messages)
    return completion


# 初始化一个 messages 数组
# messages = [
#     {
#         "role": "system",
#         "content": """你是一名身经百战的斗地主玩家，你获得了18次全国冠军，你的牌技无人能敌，现在你需要帮助一个新人，指导他的出牌技巧。""",
#     }
# ]
# assistant_output = "欢迎光临阿里云百炼手机商店，您需要购买什么尺寸的手机呢？"
# print(f"模型输出：{assistant_output}\n")
# while "我已了解您的购买意向" not in assistant_output:
#     user_input = input("请输入：")
#     # 将用户问题信息添加到messages列表中
#     messages.append({"role": "user", "content": user_input})
#     assistant_output = get_response(messages).choices[0].message.content
#     # 将大模型的回复信息添加到messages列表中
#     messages.append({"role": "assistant", "content": assistant_output})
#     print(f"模型输出：{assistant_output}")
#     print("\n")
