from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);