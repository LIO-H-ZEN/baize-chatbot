# -*- coding: utf-8 -*-
import os
import pdb
import logging
import sys
import torch

from app_modules.utils import *
from app_modules.presets import *
from app_modules.overwrites import *

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)
base_model = sys.argv[1]
adapter_model = sys.argv[2]
tokenizer, model, device = load_tokenizer_and_model(
    base_model, adapter_model, load_8bit=load_8bit
)

top_p = 0.95
temperature = 0.1
max_length_tokens = 512
max_context_length_tokens = 2048
history = ""

def predict(
    text,
    tokenizer,
    model,
    history="",
    top_p=0.95,
    temperature=0.1,
    max_length_tokens=512,
    max_context_length_tokens=2048,
):
    inputs = generate_prompt_with_history(
        text, history, tokenizer, max_length=max_context_length_tokens
    )
    if inputs is None:
        print("Input too long.")
    else:
        prompt, inputs = inputs
        begin_length = len(prompt)
    input_ids = inputs["input_ids"][:, -max_context_length_tokens:].to(device)
    torch.cuda.empty_cache()

    with torch.no_grad():
        ret = ""
        for x in sample_decode(
            input_ids,
            model,
            tokenizer,
            stop_words=["[|Human|]", "[|AI|]"],
            max_length=max_length_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            ret = x
    torch.cuda.empty_cache()
    print(prompt)
    print(x)
    print("=" * 80)

prompt = "We have a local csv file named demo_data.csv, how to use Pandas to make a summary for this csv file? (please give me complete and runable code)"
predict_once(prompt, tokenizer, model)
