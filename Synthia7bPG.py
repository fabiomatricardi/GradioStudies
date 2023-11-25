import gradio as gr
import os
from ctransformers import AutoModelForCausalLM, AutoConfig, Config #import for GGML models
import datetime

#MODEL SETTINGS also for DISPLAY
modelfile = "model\synthia-7b-v2.0-16k.Q4_K_M.gguf"
repetitionpenalty = 1.15
contextlength=16384
logfile = 'Synthia7B-GGUF-16k-promptsPlayground.txt'
print("loading model...")
stt = datetime.datetime.now()
conf = AutoConfig(Config(temperature=0.3, repetition_penalty=repetitionpenalty, batch_size=64,
                max_new_tokens=4096, context_length=contextlength))
llm = AutoModelForCausalLM.from_pretrained(modelfile,
                                        model_type="mistral", config = conf)
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
def combine(a, b, c, d):
    import datetime
    temperature = c
    max_new_tokens = d
    prompt = "SYSTEM: " + a + "\nUSER: " + b + "\nASSISTANT:\n"
    start = datetime.datetime.now()
    output = llm(prompt, 
                 temperature = temperature, 
                 repetition_penalty = 1.15, 
                 max_new_tokens=max_new_tokens)
    delta = datetime.datetime.now() - start
    logger = f"""PROMPT: \n{prompt}\nSynthia7b-16K: {output}\nGenerated in {delta}\n\n---\n\n"""
    writehistory(logger)
    generation = f"""{output} """
    return generation, delta


# MAIN GRADIO INTERFACE
with gr.Blocks(theme=gr.themes.Soft()) as demo:   #theme=gr.themes.Glass()
    #TITLE SECTION
    with gr.Row(variant='compact'):
            gr.HTML("<center>"
            + "<p>Prompt Engineering Playground - Test your favourite LLM for advanced inferences</p>"
            + "<h1>🙋‍♀️ SynthIA 16K context window</h1></center>")

    # MODEL PARAMETERS INFO SECTION
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            - **Prompt Template**: SynthIA 🙋‍♀️
            - **Repetition Penalty**: {repetitionpenalty}\n
            """)                            
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            - **Context Lenght**: {contextlength} tokens
            - **LLM Engine**: CTransformers
            """)        
        with gr.Column(scale=2):
            gr.Markdown(
            f"""
            - **Model**: {modelfile}
            - **Log File**: {logfile}
            """)
    # PLAYGROUND INTERFACE SECTION
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            ### Tunning Parameters""")
            temp = gr.Slider(label="Temperature",minimum=0.0, maximum=1.0, step=0.01, value=0.1)
            max_len = gr.Slider(label="Maximum output lenght", minimum=10,maximum=8096,step=2, value=1024)
            gr.Markdown(
            """- Fill the System Prompt and User Prompt And then click the Button below
            - For Tree of Thoughts use *Elaborate on the topic using a Tree of Thoughts and backtrack when necessary to construct a clear, cohesive Chain of Thought reasoning. Always answer without hesitation.*
            """)
            btn = gr.Button(value="🙋‍♀️ Generate")
            gentime = gr.Textbox(value="", label="Generation Time:")
        with gr.Column(scale=4):
            txt = gr.Textbox(label="System Prompt", lines=3)
            txt_2 = gr.Textbox(label="User Prompt", lines=5)
            txt_3 = gr.Textbox(value="", label="Output", lines = 10, show_copy_button=True)
            btn.click(combine, inputs=[txt, txt_2,temp,max_len], outputs=[txt_3,gentime])

if __name__ == "__main__":
    demo.launch(inbrowser=True)