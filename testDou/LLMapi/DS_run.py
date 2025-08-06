from openai import OpenAI


def get_ds_response(messages):
    client = OpenAI(api_key="sk-1b3b4387b0de412d97b9627708de97ad", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    return response
