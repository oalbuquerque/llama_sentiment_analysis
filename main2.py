# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import pyarrow as pa

from pathlib import Path

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def load(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    arrow_dir = Path(ckpt_dir).expanduser() / 'arrow'

    if not arrow_dir.exists():
        print('Converting checkpoints to arrow format')
        checkpoints = sorted(Path(ckpt_dir).expanduser().glob("*.pth"))
        for ckpt_file in checkpoints:
            print(ckpt_file)
            index = ckpt_file.parts[-1].split('.')[-2]

            ckpt = torch.load(ckpt_file, map_location='cpu')
            (arrow_dir / index).mkdir(parents=True, exist_ok=True)
            for k, v in ckpt.items():
                tens = pa.Tensor.from_numpy(v.numpy())
                with pa.output_stream(arrow_dir / index / k) as f:
                    pa.ipc.write_tensor(tens, f)
            ckpt = None

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    print("Loading checkpoint")
    segments = sorted((arrow_dir / '00').glob("*"))
    # print(segments)

    checkpoint = {}
    files = []
    for seg in segments:
        f = pa.memory_map(str(seg))
        files.append(f)
        t = pa.ipc.read_tensor(f).to_numpy()
        t = torch.from_numpy(t)
        checkpoint[seg.parts[-1]] = t

    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    torch.set_default_tensor_type(torch.BFloat16Tensor)
    # torch.set_default_tensor_type(torch.FloatTensor)

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    print("Loading tokenizer")
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    print("Loading model")
    model = Transformer(model_args)

    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    model.load_state_dict(torch.load(checkpoints[-1]), strict=False)

    for f in files:
        f.close()
    files = None

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.8,
        top_p: float = 0.95,  # use 0.95 or so for top_p sampler, and 0.0 for top_k sampler
        top_k: int = 40,
        repetition_penalty: float = (1.0 / 0.85),  # 1.0 to disable repetition_penalty
        sampler: str = 'top_p',  # top_p or top_k
        max_seq_len: int = 2048,
        max_batch_size: int = 1,
):
    generator = load(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)

    prompts = """Dado um conjunto de tweets em português, preprocessados e limpos (sem menções, retweets, hashtags e pontuações), realize a análise do sentimento dos textos,
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

#prompts = [
# "I believe the meaning of life is",
#"""Write the Python code with detailed comments to generate 256 random integers in the range from -128 to 512, inclusive.
#\\begin{code}\n""",
#]
    
    results = generator.generate(
        prompts, max_gen_len=max_seq_len, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, sampler=sampler
    )

    for result in results:
        print("\n==================================\n")
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
