from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    prompts = [
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
    ]

    prompts = [
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
        Tweets:"mas será mesmo que preciso saber criar inteligência artificial para criar um chatbot.";"google emite 'sinal de alerta' após lançamento do chatgpt.";"pra parecer história de filme você vai ter que procurar em todas as redes sociais e apps de reconhecimento facial.";"os chatbots podem entregar a seu cliente exatamente o que ele quer apenas batendo um papo com ele acesse."
        Sentimentos:""",
    ]
   
    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
