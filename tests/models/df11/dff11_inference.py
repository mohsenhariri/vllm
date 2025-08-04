import os
import sys
import torch
from vllm import LLM, SamplingParams

print(f"CUDA Version: {torch.version.cuda}") # tested only with CUDA 12.6


# regular models

# model_path = "/mnt/rds/VipinRDS/VipinRDS/users/mxh1029/llms/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"

# llm = LLM(
#     model=model_path,
#     enforce_eager=True,
#     model_impl="transformers"

# )


df_model_path = "DF11/weights/llama3.1-8b-it-df11"


llm = LLM(
    model=df_model_path,
    enforce_eager=True,
    load_format="df11", # by setting this argument, it loaded `DF11ModelLoader`
    model_impl="transformers",
    dtype="bfloat16"  

)


prompts = "What is the meaning of life?"


sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=500,
)

outputs = llm.generate(prompts, sampling_params=sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 50)
