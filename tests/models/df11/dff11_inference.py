import torch
import os

os.environ["VLLM_USE_V1"] = "1"

from vllm import LLM, SamplingParams

print(f"CUDA Version: {torch.version.cuda}") # tested only with CUDA 12.6


df_model_path = "DF11/weights/llama3.1-8b-it-df11"

# llm = LLM(
#     model=df_model_path,
#     enforce_eager=True,
#     load_format="df11", # by setting this, `DF11ModelLoader` is loaded
#     model_impl="transformers",
#     dtype="bfloat16"  
# )

llm = LLM(
    model=df_model_path,
    load_format="df11", # by setting this, `DF11ModelLoader` is loaded
    dtype="bfloat16"  
)

prompts = "What is the meaning of life?"


sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=5,
)

outputs = llm.generate(prompts, sampling_params=sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 50)
