import gradio as gr
import os
from ctransformers import AutoModelForCausalLM, AutoConfig, Config #import for GGML models
import datetime

temperature = 0.32 
max_new_tokens=1100
modelfile = "models/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf"
repetitionpenalty = 1.15
contextlength=62048
logfile = 'TinyLlamaOpenOrca1.1B-stream.txt'
print("loading model...")
stt = datetime.datetime.now()
conf = AutoConfig(Config(temperature=0.3, repetition_penalty=repetitionpenalty, batch_size=64,
                max_new_tokens=4096, context_length=contextlength))
llm = AutoModelForCausalLM.from_pretrained(modelfile,
                                        model_type="llama",config = conf) #model_type="stablelm", 
dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")
#MODEL SETTINGS also for DISPLAY

def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        writehistory(f"USER: {user_message}")
        return "", history + [[user_message, None]]

    def bot(history):
        SYSTEM_PROMPT = """<|im_start|>system
        You are a helpful bot. Your answers are clear and concise.
        <|im_end|>

        """    
        prompt = f"<|im_start|>system<|im_end|><|im_start|>user\n{history[-1][0]}<|im_end|>\n<|im_start|>assistant\n"  
        print(f"history lenght: {len(history)}")
        if len(history) == 1:
            print("this is the first round")
        else:
            print("here we should pass more conversations")
        history[-1][1] = ""
        for character in llm(prompt, 
                 temperature = temperature, 
                 repetition_penalty = 1.15, 
                 max_new_tokens=max_new_tokens,
                 stop = ['<|im_end|>'],
                 stream = True):
            history[-1][1] += character
            yield history
        writehistory(f"BOT: {history}")    

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.queue()
demo.launch(inbrowser=True)

