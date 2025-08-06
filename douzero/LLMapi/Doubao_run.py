import os
from openai import OpenAI


def get_db_response(messages):
    client = OpenAI(
        # 从环境变量中读取您的方舟API Key
        api_key="88483755-69cd-43dc-b94a-7c6200af0729",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    completion = client.chat.completions.create(
        # 将推理接入点 <Model>替换为 Model ID
        model="doubao-seed-1.6-250615",
        messages=messages
    )
    return completion
