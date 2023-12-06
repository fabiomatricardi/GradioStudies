from llama_cpp import Llama
from rich.console import Console
import datetime
from rich.prompt import Prompt
console = Console(width=110) #initialize RICH console 
console.clear()  #clear screen


# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
  model_path="model/starling-lm-7b-alpha.Q4_K_M.gguf",  # Download the model file first
  n_ctx=8192,  # The max sequence length to use - note that longer sequence lengths require much more resources
  #n_threads=2,            # The number of CPU threads to use, tailor to your system and the resulting performance
)
console.clear()  #clear screen
console.print("[bold italic bright_red]Model [bold bright_yellow]starling-lm-7b-alpha.Q4_K_M.gguf [bold italic bright_red]Loaded")
user = Prompt.ask("[bold green1 reverse]User: ")
prompt = f"GPT4 User: {user}<|end_of_turn|>GPT4 Assistant:"
# Simple inference example
start = datetime.datetime.now()
output = llm(prompt, max_tokens=912, stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
            echo=False, stream=True)

for chunk in output:
    console.print(f'[bold green1]{chunk["choices"][0]["text"]}', end="")
delta = datetime.datetime.now() - start
console.rule()
console.print(f"[bold green]Completed in {delta}")