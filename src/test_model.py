from transformers import pipeline

# A small, free instruction-tuned model that runs on a normal laptop
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
)

messages = [
    {"role": "system", "content": "You are a helpful compliance assistant."},
    {"role": "user", "content": "In one sentence, what is the purpose of UK GDPR?"},
]

result = generator(messages, max_new_tokens=100)
print(result[0]["generated_text"][-1]["content"])