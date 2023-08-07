# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire
import pandas as pd

from generation import Llama

def tweets():
    df = pd.read_csv('sentiment_analysis.csv', sep=';')
    tweets_cleaned = "".join(f"{index + 1}-'{content}';" for index, content in enumerate(df['clean_text'].astype(str)))
    tweets_sentiment = {"role": "user", "content": tweets_cleaned}
    return tweets_sentiment

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
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
    [{"role": "system", "content": "Dada uma coleção de tweets em português previamente processados e limpos (excluindo menções, retweets, hashtags e pontuações), realize a análise de sentimento desses textos, classificando-os como positivo, negativo ou neutro. Entregue somente o resultado do sentimento para cada tweet, sem fazer menção ao comando dado ou incluir explicações adicionais. Os resultados devem ser fornecidos na mesma ordem dos tweets originais separados por ponto e vírgula ';'."},
    {'role': 'user', 'content': "1-'acho muito bacana o whatsapp só permitir leitura de qrcode pro whatsapp web depois de fazer reconhecimento facial';2-'o facebook vem aprimorando a sua tecnologia de reconhecimento facial possibilitando que deficientes visuais consigam ouvir uma descrição de quem está na foto mesmo que o amigo não esteja marcado na publicação saiba mais';3-'órgãos de proteção de dados da europa querem banir reconhecimento facial';4-'o conselho europeu para a proteção de dados e a autoridade europeia para a proteção de dados defendem que a proibição do uso de sistemas de reconhecimento facial em espaços públicos é fundamental para preservar os direitos e liberdades dos cidadãos da ue';5-'univeidade usa reconhecimento facial para controlar frequência dos alunos';6-'usei o chat gpt e estou com medo da capacidade desse monstro';7-'google emite “sinal de alerta” após lançamento do chatgpt';8-'primeiro que plugar um chatbot ganha';9-'como desenvolver um chatbot aprenda sobre essa tecnologia na maratona bots que começa no dia de janeiro conheça';10-'google e intel lançam kit de inteligência artificial para reconhecimento de objetos';11-'chatbot a evolução no atendimento ao cliente';12-'ellevo lança chatbot e app em nova versão da plataforma de atendimento';13-'posso parecer jurássico mas minha intuição analógica diz que o mundo deveria regular fortemente o avanço da tecnologia de reconhecimento facial por aplicativos e robôs é uma armadilha claramente distópica só acho';14-'chatbots para negócios monte seu chatbot em horas ou menos sem precisar saber programar';15-'chineses estão desenvolvendo gps e reconhecimento facial para galinhas';16-'os chatbots podem entregar a seu cliente exatamente o que ele quer apenas batendo um papo com ele acesse';17-'o vídeo mostra um futuro em que os drones autônomos de tamanho de palma usaram tecnologia de reconhecimento facial e explosivos a bordo para cometer massacres';18-'defeito inesperado iphone x começa a causar dor em seus usuários causa queixas de usuários nomeadamente surgiram informações sobre problemas com tela reconhecimento facial face id entre outras falhas além disso no fim do ano passado a app';19-'mas será mesmo que preciso saber criar inteligência artificial para criar um chatbot';20-'maislidasbuzzmonitor como um chatbot pode impulsionar sua estratégia de marketing digital descubra aqui';21-'chatbot realiza dos atendimentos do bb em rede social';22-'nem só de reconhecimento de padrões vive as aplicações de ia automatização de serviços podem aperfeiçoar bastante os processos de uma organização reduzindo os custos e aumentando a satisfação do cliente';23-'o pesadelo da invasão de privacidade até onde vai a tecnologia do reconhecimento facial';24-'deve haver um sistema avançado de monitorização por vídeo vigilância e reconhecimento facial';25-'gostei de um vídeo chatbot perguntas e respostas por voz criando uma ia pelo';26-'o reconhecimento facial está chegando ao varejo o que preocupa é que através de qualquer câmera da empresa que registrou nossa face é possível nos reconhecer e registrar tudo desde o momento que entrarmos no estabelecimento estão preparados para isso';27-'será que o chatgpt vai substituir o google';28-'os vieses nos modelos de texto ai serão um grande problema por exemplo a ética embutida no chatgpt permite brincar com o comunismo ao mesmo tempo em que condena claramente o nazismo tão desrespeitoso com as vítimas do terror vermelho e errado';29-'o chatbot vai fazer sua própria programação';30-'será que um speech to text não te ajuda a dar vazão enquanto não acha alguém pra te ajudar nessa task';31-'é machine learning pra trabalhar com natural language processing';32-'a tecnologia bot to bot é um novo modelo de interação entre bots em que um bot principal identifica a solicitação de um cliente e delega a tarefa a um bot especializado dentro da área específica chatbot';"},
    ],
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
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
            {"role": "user", "content": "1-'acho muito bacana o whatsapp só permitir leitura de qrcode pro whatsapp web depois de fazer reconhecimento facial';2-'o facebook vem aprimorando a sua tecnologia de reconhecimento facial possibilitando que deficientes visuais consigam ouvir uma descrição de quem está na foto mesmo que o amigo não esteja marcado na publicação saiba mais';3-'órgãos de proteção de dados da europa querem banir reconhecimento facial';4-'o conselho europeu para a proteção de dados e a autoridade europeia para a proteção de dados defendem que a proibição do uso de sistemas de reconhecimento facial em espaços públicos é fundamental para preservar os direitos e liberdades dos cidadãos da ue';5-'univeidade usa reconhecimento facial para controlar frequência dos alunos';6-'usei o chat gpt e estou com medo da capacidade desse monstro';7-'google emite “sinal de alerta” após lançamento do chatgpt';8-'primeiro que plugar um chatbot ganha';9-'como desenvolver um chatbot aprenda sobre essa tecnologia na maratona bots que começa no dia de janeiro conheça';10-'google e intel lançam kit de inteligência artificial para reconhecimento de objetos';11-'chatbot a evolução no atendimento ao cliente';12-'ellevo lança chatbot e app em nova versão da plataforma de atendimento';13-'posso parecer jurássico mas minha intuição analógica diz que o mundo deveria regular fortemente o avanço da tecnologia de reconhecimento facial por aplicativos e robôs é uma armadilha claramente distópica só acho';14-'chatbots para negócios monte seu chatbot em horas ou menos sem precisar saber programar';15-'chineses estão desenvolvendo gps e reconhecimento facial para galinhas';16-'os chatbots podem entregar a seu cliente exatamente o que ele quer apenas batendo um papo com ele acesse';17-'o vídeo mostra um futuro em que os drones autônomos de tamanho de palma usaram tecnologia de reconhecimento facial e explosivos a bordo para cometer massacres';18-'defeito inesperado iphone x começa a causar dor em seus usuários causa queixas de usuários nomeadamente surgiram informações sobre problemas com tela reconhecimento facial face id entre outras falhas além disso no fim do ano passado a app';19-'mas será mesmo que preciso saber criar inteligência artificial para criar um chatbot';20-'maislidasbuzzmonitor como um chatbot pode impulsionar sua estratégia de marketing digital descubra aqui';21-'chatbot realiza dos atendimentos do bb em rede social';22-'nem só de reconhecimento de padrões vive as aplicações de ia automatização de serviços podem aperfeiçoar bastante os processos de uma organização reduzindo os custos e aumentando a satisfação do cliente';23-'o pesadelo da invasão de privacidade até onde vai a tecnologia do reconhecimento facial';24-'deve haver um sistema avançado de monitorização por vídeo vigilância e reconhecimento facial';25-'gostei de um vídeo chatbot perguntas e respostas por voz criando uma ia pelo';26-'o reconhecimento facial está chegando ao varejo o que preocupa é que através de qualquer câmera da empresa que registrou nossa face é possível nos reconhecer e registrar tudo desde o momento que entrarmos no estabelecimento estão preparados para isso';27-'será que o chatgpt vai substituir o google';28-'os vieses nos modelos de texto ai serão um grande problema por exemplo a ética embutida no chatgpt permite brincar com o comunismo ao mesmo tempo em que condena claramente o nazismo tão desrespeitoso com as vítimas do terror vermelho e errado';29-'o chatbot vai fazer sua própria programação';30-'será que um speech to text não te ajuda a dar vazão enquanto não acha alguém pra te ajudar nessa task';31-'é machine learning pra trabalhar com natural language processing';32-'a tecnologia bot to bot é um novo modelo de interação entre bots em que um bot principal identifica a solicitação de um cliente e delega a tarefa a um bot especializado dentro da área específica chatbot';"},
        ],
    ]
    
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
