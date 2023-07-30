import torch

#!pip install transformers
#from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

# Download the Model Weight in the following form: https://ai.meta.com/resources/models-and-libraries/llama-downloads/

"""
After dowloading the weights, you should be able to convert your own model, or use third-party models like the HuggingFaces trasnformers format using the conversion script bellow:
--------------------------------------------------------------------------------------------
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
--------------------------------------------------------------------------------------------
"""

"""
Or you can define the paths of the checkpoints already created
--------------------------------------------------------------
model_name = 'alpaca'
path = '/mnt/database/llama-main/'
checkpoint = path + model_name
--------------------------------------------------------------
"""

#tokenizer = LlamaTokenizer.from_pretrained("/output/path") # PATH_TO_CONVERTED_WEIGHTS
#model = LlamaForCausalLM.from_pretrained("/output/path") # PATH_TO_CONVERTED_TOKENIZER

tokenizer = AutoTokenizer.from_pretrained("/home/gerontech/llama2/models/tokenizer.model") # PATH_TO_CONVERTED_TOKENIZER
model = AutoModelForCausalLM.from_pretrained("/home/gerontech/llama2/models/7B", device_map="auto") # PATH_TO_CONVERTED_WEIGHTS


prompt = """Dado um conjunto de tweets em português, preprocessados e limpos (sem menções, retweets, hashtags e pontuações), realize a análise do sentimento dos textos,
        classificando-os em positivo, negativo ou neutro. Forneça apenas o resultado do sentimento para cada tweet, sem menções ao comando dado ou explicações adicionais.
        Os resultados devem ser entregues na mesma ordem das perguntas. O formato das respostas deve ser adequado para inclusão em um dataframe no Python:

        o pesadelo da invasão de privacidade até onde vai a tecnologia do reconhecimento facial => negativo;
        os vieses nos modelos de texto ai serão um grande problema por exemplo a ética embutida no chatgpt permite brincar com o comunismo ao mesmo tempo em que condena claramente o nazismo tão
        desrespeitoso com as vítimas do terror vermelho e errado => negativo;
        será que o chatgpt vai substituir o google => neutro;
        deve haver um sistema avançado de monitorização por vídeo vigilância e reconhecimento facial => neutro;
        o facebook vem aprimorando a sua tecnologia de reconhecimento facial possibilitando que deficientes visuais consigam ouvir uma descrição de quem está na foto mesmo que o amigo não esteja marcado
        na publicação saiba mais => positivo;
        nem só de reconhecimento de padrões vive as aplicações de ia automatização de serviços podem aperfeiçoar bastante os processos de uma organização reduzindo os custos e aumentando a satisfação do cliente => positivo;
        mas será mesmo que preciso saber criar inteligência artificial para criar um chatbot => ;
        google emite “sinal de alerta” após lançamento do chatgpt => ;
        pra parecer história de filme você vai ter que procurar em todas as redes sociais e apps de reconhecimento facial => ;
        os chatbots podem entregar a seu cliente exatamente o que ele quer apenas batendo um papo com ele acesse => ?;
        """


inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=64)
text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(text)
