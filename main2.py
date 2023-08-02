import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = alpaca'
path = '/mnt/database/llama-main/'
checkpoint = path + model_name

model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prompt = "What Brazilian researchers won the Nobel Prize?"
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=20)
text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(text)
