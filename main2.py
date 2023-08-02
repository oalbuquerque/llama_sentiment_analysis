import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = '7B'
path = '/home/gerontech/llama2/models/'
checkpoint = path + model_name

#model = LlamaForCausalLM.from_pretrained("/home/gerontech/llama2/models/7B") # PATH_TO_CONVERTED_TOKENIZER
#tokenizer = LlamaTokenizer.from_pretrained("/home/gerontech/llama2/models/tokenizer.model") # PATH_TO_CONVERTED_WEIGHTS


model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prompt = "What Brazilian researchers won the Nobel Prize?"
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=20)
text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(text)
