from typing import Tuple
from typing import Optional
import os
import sys
import torch
import fire
import time
import json
import pandas as pd

from pathlib import Path
from generation import Llama
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

def tweets():
    df = pd.read_csv('sentiment_analysis.csv', sep=';')
    tweets_cleaned = "".join(f"{index + 1}-'{content}';" for index, content in enumerate(df['clean_text'][:5].astype(str)))
    tweets_sentiment = 'Tweets:'+tweets_cleaned
    return tweets_sentiment

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 2000,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = 512,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts2 = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
        Sentiment: Negative
        ###
        Tweet: "My day has been 👍"
        Sentiment: Positive
        ###
        Tweet: "This is the link to the article"
        Sentiment: Neutral
        ###
        Tweet: "This new music video was incredibile"
        Sentiment:""",
                """Translate English to French:
        
        sea otter => loutre de mer
        
        peppermint => menthe poivrée
        
        plush girafe => girafe peluche
        
        cheese =>""",
        
        # sentiment polarity analysis
        """Tweet:"o pesadelo da invasão de privacidade até onde vai a tecnologia do reconhecimento facial."
        Sentimento: negativo
        ###
        Tweet:"os vieses nos modelos de texto ai serão um grande problema por exemplo a ética embutida no chatgpt permite brincar com o comunismo ao mesmo tempo em que condena claramente o nazismo tão
        desrespeitoso com as vítimas do terror vermelho e errado."
        Sentimento:negativo
        ###
        Tweet:"será que o chatgpt vai substituir o google"
        Sentimento: neutro
        ###
        Tweet:"deve haver um sistema avançado de monitorização por vídeo vigilância e reconhecimento facial."
        Sentimento:neutro
        ###
        Tweet:"o facebook vem aprimorando a sua tecnologia de reconhecimento facial possibilitando que deficientes visuais consigam ouvir uma descrição de quem está na foto mesmo que o amigo não esteja marcado
        na publicação saiba mais."
        Sentimento:positivo
        ###
        Tweet:"nem só de reconhecimento de padrões vive as aplicações de ia automatização de serviços podem aperfeiçoar bastante os processos de uma organização reduzindo os custos e aumentando a satisfação do cliente."
        Sentimento:positivo
        ###
        tweets(df['clean_text'])""",
    ]

    prompts = [
        # sentiment polarity analysis
        f"""Tweet: "reconhecimento facial é vida"
        Sentiment: positivo
        ###
        Tweet: "essa é uma das iniciativas mais legais que vi nos últimos tempos experiência interativa com chatbot simula um futuro sem a amazônia e mobiliza jovens na atualidade"
        Sentiment: positivo
        ###
        Tweet: "polícia militar está utilizando em fase de teste câmeras de videomonitoramento que com a base de dados do estado utiliza reconhecimento facial para cumprir mandados de prisões em aberto entre outras ocorrências polícia"
        Sentiment: neutro
        ###
        Tweet: "pesquisadores usam visão computacional para reconstruir objetos"
        Sentiment: neutro
        ###
        Tweet: "e na inglaterra falta educação e punição na era de inteligência artificial e reconhecimento facial o sujeito que briga consegue retornar para o estádio é incrível a falta de segurança e de organização"
        Sentiment: negativo
        ###
        Tweet: "e mesmo assim o serviço que vocês entregam é um lixo entra ano sai ano e vocês não fazem nada para melhorar o chatbot de vocês é genérico e ineficiente o chat web a mesma coisa ligar é pedir pra passar nervoso na espera"
        Sentiment: negativo
        ###
        {tweets()}
        Sentiments: """
    ]

    prompts3 = [
        # sentiment polarity analysis
        f"""Tweets:1-"o processo de captação e retenção de alunos tem muito a ganhar com a utilização de chatbots com inteligência artificial";2-"essa é uma das iniciativas mais legais que vi nos últimos tempos experiência interativa com chatbot simula um futuro sem a amazônia e mobiliza jovens na atualidade";3-"polícia militar está utilizando em fase de teste câmeras de videomonitoramento que com a base de dados do estado utiliza reconhecimento facial para cumprir mandados de prisões em aberto entre outras ocorrências polícia";4-"pesquisadores usam visão computacional para reconstruir objetos";5-"e na inglaterra falta educação e punição na era de inteligência artificial e reconhecimento facial o sujeito que briga consegue retornar para o estádio é incrível a falta de segurança e de organização";6-"e mesmo assim o serviço que vocês entregam é um lixo entra ano sai ano e vocês não fazem nada para melhorar o chatbot de vocês é genérico e ineficiente o chat web a mesma coisa ligar é pedir pra passar nervoso na espera";
        Sentiments:1-positivo;2-positivo;3-neutro;4-neutro;5-negativo;6-negativo
        ###
        {tweets()}
        Sentiments: """,
    ]
   
    results = generator.generate(
        prompts3, max_gen_len=1200, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
