import gradio as gr
from ctransformers import AutoModelForCausalLM, AutoConfig, Config #import for GGML models
import datetime
from threading import Thread
#MODEL SETTINGS also for DISPLAY

convHistory = ''
modelfile = "model/open-llama-3b-v2-wizard-evol-instuct-v2-196k.Q4_K_M.gguf"
repetitionpenalty = 1.15
contextlength=62048
logfile = 'Openllama3B-GGML-promptsPlayground.txt'
print("loading model...")
stt = datetime.datetime.now()
conf = AutoConfig(Config(temperature=0.3, repetition_penalty=repetitionpenalty, batch_size=64,
                max_new_tokens=4096, context_length=contextlength))
model = AutoModelForCausalLM.from_pretrained(modelfile,
                                        model_type="llama",config = conf) #model_type="stablelm", 
dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")



def predict(message, history):

    history_transformer_format = history + [[message, ""]]

    messages = "".join(["".join(["\n### HUMAN:\n"+item[0], "\n\n### RESPONSE:\n"+item[1]])  #curr_system_message +
                for item in history_transformer_format])

    return model(messages, 
                 temperature = 0.3, 
                 repetition_penalty = 1.15, 
                 max_new_tokens=2048)


gr.ChatInterface(predict).launch()