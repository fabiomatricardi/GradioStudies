import gradio as gr
from llama_cpp import Llama
import datetime

#MODEL SETTINGS also for DISPLAY
convHistory = ''
modelfile = "model/starling-lm-7b-alpha.Q4_K_M.gguf"
repetitionpenalty = 1.15
contextlength=8192
logfile = 'StarlingLM7B_logs.txt'
print("loading model...")
stt = datetime.datetime.now()
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
  model_path=modelfile,  # Download the model file first
  n_ctx=contextlength,  # The max sequence length to use - note that longer sequence lengths require much more resources
  #n_threads=2,            # The number of CPU threads to use, tailor to your system and the resulting performance
)
dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")

def writehistory(text):
    with open(logfile, 'a') as f:
        f.write(text)
        f.write('\n')
    f.close()

"""
gr.themes.Base()
gr.themes.Default()
gr.themes.Glass()
gr.themes.Monochrome()
gr.themes.Soft()
"""
def combine(a, b, c, d,e,f):
    global convHistory
    import datetime
    SYSTEM_PROMPT = f"""{a}


    """ 
    # parameters here: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__       
    temperature = c
    max_new_tokens = d
    repeat_penalty = f
    top_p = e
    prompt = f"GPT4 User: {b}<|end_of_turn|>GPT4 Assistant:"
    start = datetime.datetime.now()
    generation = ""
    delta = ""
    prompt_tokens = f"Prompt Tokens: {len(llm.tokenize(bytes(prompt,encoding='utf-8')))}"
    generated_text = ""
    answer_tokens = ''
    total_tokens = ''   
    for character in llm(prompt, 
                max_tokens=max_new_tokens, 
                stop=["</s>"],
                temperature = temperature,
                repeat_penalty = repeat_penalty,
                top_p = top_p,   # Example stop token - not necessarily correct for this specific model! Please check before using.
                echo=False, 
                stream=True):
        generation += character["choices"][0]["text"]

        answer_tokens = f"Out Tkns: {len(llm.tokenize(bytes(generation,encoding='utf-8')))}"
        total_tokens = f"Total Tkns: {len(llm.tokenize(bytes(prompt,encoding='utf-8'))) + len(llm.tokenize(bytes(generation,encoding='utf-8')))}"
        delta = datetime.datetime.now() - start
        yield generation, delta, prompt_tokens, answer_tokens, total_tokens
    timestamp = datetime.datetime.now()
    logger = f"""time: {timestamp}\n Temp: {temperature} - MaxNewTokens: {max_new_tokens} - RepPenalty: 1.5 \nPROMPT: \n{prompt}\nClaude2Alpaca-7B: {generation}\nGenerated in {delta}\nPromptTokens: {prompt_tokens}   Output Tokens: {answer_tokens}  Total Tokens: {total_tokens}\n\n---\n\n"""
    writehistory(logger)
    convHistory = convHistory + prompt + "\n" + generation + "\n"
    print(convHistory)
    return generation, delta, prompt_tokens, answer_tokens, total_tokens    
    #return generation, delta


# MAIN GRADIO INTERFACE
with gr.Blocks(theme='WeixuanYuan/Soft_dark') as demo:   #theme=gr.themes.Glass()  #theme='remilia/Ghostly'
    #TITLE SECTION
    with gr.Row(variant='compact'):
            with gr.Column(scale=12):
                gr.HTML("<center>"
                + "<h3>Prompt Engineering Playground!</h3>"
                + "<h1>🐦 StarlingLM-7b 8K context window</h2></center>")  
            gr.Image(value='./starling.png', height=95, show_label = False, 
                     show_download_button = False, container = False)    
    # INTERACTIVE INFOGRAPHIC SECTION
    with gr.Row():
        with gr.Column(min_width=80):
            gentime = gr.Textbox(value="", placeholder="Generation Time:", min_width=50, show_label=False)                          
        with gr.Column(min_width=80):
            prompttokens = gr.Textbox(value="", placeholder="Prompt Tkn:", min_width=50, show_label=False)
        with gr.Column(min_width=80):
            outputokens = gr.Textbox(value="", placeholder="Output Tkn:", min_width=50, show_label=False)            
        with gr.Column(min_width=80):
            totaltokens = gr.Textbox(value="", placeholder="Total Tokens:", min_width=50, show_label=False)  

    # PLAYGROUND INTERFACE SECTION
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            ### Tunning Parameters""")
            temp = gr.Slider(label="Temperature",minimum=0.0, maximum=1.0, step=0.01, value=0.42)
            top_p = gr.Slider(label="Temperature",minimum=0.0, maximum=1.0, step=0.01, value=0.8)
            repPen = gr.Slider(label="Temperature",minimum=0.0, maximum=4.0, step=0.01, value=1.2)
            max_len = gr.Slider(label="Maximum output lenght", minimum=10,maximum=7000,step=2, value=900)
            gr.Markdown(
            """
            Fill the System Prompt and User Prompt
            And then click the Button below
            """)
            btn = gr.Button(value="🐦 Generate", variant='primary')
            gr.Markdown(
            f"""
            - **Prompt Template**: OpenChat 🐦
            - **Repetition Penalty**: {repetitionpenalty}
            - **Context Lenght**: {contextlength} tokens
            - **LLM Engine**: CTransformers
            - **Model**: 🐦 StarlingLM-7b
            - **Log File**: {logfile}
            """) 


        with gr.Column(scale=4):
            txt = gr.Textbox(label="System Prompt", value = "", placeholder = "This models does not have any System prompt...",lines=1, interactive = False)
            txt_2 = gr.Textbox(label="User Prompt", lines=6)
            txt_3 = gr.Textbox(value="", label="Output", lines = 13, show_copy_button=True)
            btn.click(combine, inputs=[txt, txt_2,temp,max_len,top_p,repPen], outputs=[txt_3,gentime,prompttokens,outputokens,totaltokens])


if __name__ == "__main__":
    demo.launch(inbrowser=True)