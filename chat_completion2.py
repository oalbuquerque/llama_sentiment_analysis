# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire
import pandas as pd

from generation import Llama

def tweets():
    df = pd.read_csv('sentiment_analysis.csv', sep=';')
    tweets_cleaned = "".join(f"{index + 1}-{content};" for index, content in enumerate(df['clean_text'][:1].astype(str)))
    tweets_sentiment = {"role": "user", "content": f"""{tweets_cleaned}"""}
    return tweets_sentiment
    
print(tweets())

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs2 = [
                {"role": "system", "content":"Dada uma coleção de tweets em português previamente processados e limpos (excluindo menções, retweets, hashtags e pontuações), realize a análise de sentimento desses textos, classificando-os como positivo, negativo ou neutro. Entregue somente o resultado do sentimento para cada tweet, sem fazer menção ao comando dado ou incluir explicações adicionais. Os resultados devem ser fornecidos na mesma ordem dos tweets originais separados por ponto e vírgula ';'."},
                {'role': 'user', 'content': "1-'acho muito bacana o whatsapp só permitir leitura de qrcode pro whatsapp web depois de fazer reconhecimento facial';2-'o facebook vem aprimorando a sua tecnologia de reconhecimento facial possibilitando que deficientes visuais consigam ouvir uma descrição de quem está na foto mesmo que o amigo não esteja marcado na publicação saiba mais';3-'órgãos de proteção de dados da europa querem banir reconhecimento facial';4-'o conselho europeu para a proteção de dados e a autoridade europeia para a proteção de dados defendem que a proibição do uso de sistemas de reconhecimento facial em espaços públicos é fundamental para preservar os direitos e liberdades dos cidadãos da ue';5-'univeidade usa reconhecimento facial para controlar frequência dos alunos';6-'usei o chat gpt e estou com medo da capacidade desse monstro';7-'google emite “sinal de alerta” após lançamento do chatgpt';8-'primeiro que plugar um chatbot ganha';9-'como desenvolver um chatbot aprenda sobre essa tecnologia na maratona bots que começa no dia de janeiro conheça';10-'google e intel lançam kit de inteligência artificial para reconhecimento de objetos';"},
    ]
    
    dialogs = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content":"Dada uma coleção de tweets em português previamente processados e limpos (excluindo menções, retweets, hashtags e pontuações), realize a análise de sentimento desses textos, classificando-os como positivo, negativo ou neutro. Entregue somente o resultado do sentimento para cada tweet, sem fazer menção ao comando dado ou incluir explicações adicionais. Os resultados devem ser fornecidos na mesma ordem dos tweets originais separados por ponto e vírgula ';'."},
            {'role': 'user', 'content': "1-'acho muito bacana o whatsapp só permitir leitura de qrcode pro whatsapp web depois de fazer reconhecimento facial';2-'o facebook vem aprimorando a sua tecnologia de reconhecimento facial possibilitando que deficientes visuais consigam ouvir uma descrição de quem está na foto mesmo que o amigo não esteja marcado na publicação saiba mais';3-'órgãos de proteção de dados da europa querem banir reconhecimento facial';4-'o conselho europeu para a proteção de dados e a autoridade europeia para a proteção de dados defendem que a proibição do uso de sistemas de reconhecimento facial em espaços públicos é fundamental para preservar os direitos e liberdades dos cidadãos da ue';5-'univeidade usa reconhecimento facial para controlar frequência dos alunos';6-'usei o chat gpt e estou com medo da capacidade desse monstro';7-'google emite “sinal de alerta” após lançamento do chatgpt';8-'primeiro que plugar um chatbot ganha';9-'como desenvolver um chatbot aprenda sobre essa tecnologia na maratona bots que começa no dia de janeiro conheça';10-'google e intel lançam kit de inteligência artificial para reconhecimento de objetos';"},
        ],
    ]
    
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    print(results)
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
