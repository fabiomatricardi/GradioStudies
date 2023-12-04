import gradio as gr
from ctransformers import AutoModelForCausalLM, AutoConfig, Config #import for GGML models
import datetime
from threading import Thread
#MODEL SETTINGS also for DISPLAY

convHistory = ''
modelfile = "models/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf"
repetitionpenalty = 1.15
contextlength=62048
logfile = 'TinyLlama1Bopenorca-GGUF-chatLog.txt'
print("loading model...")
stt = datetime.datetime.now()
conf = AutoConfig(Config(temperature=0.3, repetition_penalty=repetitionpenalty, batch_size=64,
                max_new_tokens=1200, context_length=contextlength))
model = AutoModelForCausalLM.from_pretrained(modelfile,
                                        model_type="llama",config = conf) #model_type="stablelm", 
dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")

def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

#"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"

SYSTEM_PROMPT = """<|im_start|>system
You are a helpful bot. Your answers are clear and concise.
<|im_end|>

"""

# Formatting function for message and history
def format_message(message: str, history: list, memory_limit: int = 4) -> str:
    """
    Formats the message and history for the Llama model.

    Parameters:
        message (str): Current message to send.
        history (list): Past conversation history.
        memory_limit (int): Limit on how many past interactions to consider.

    Returns:
        str: Formatted message string
    """
    # always keep len(history) <= memory_limit
    if len(history) > memory_limit:
        history = history[-memory_limit:]

    if len(history) == 0:
        return SYSTEM_PROMPT + f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant"

    formatted_message = SYSTEM_PROMPT + f"{history[0][0]} <|im_end|>\n<|im_start|>assistant\n{history[0][1]}"

    # Handle conversation history
    for user_msg, model_answer in history[1:]:
        formatted_message += f"<|im_start|>user\n{user_msg} <|im_end|>\n<|im_start|>assistant\n{model_answer}\n"

    # Handle the current message
    formatted_message += f"<|im_start|>user\n{message} <|im_end|>\n<|im_start|>assistant\n"

    return formatted_message

# Generate a response from the Llama model
def get_tinyllama_response(message: str, history: list) -> str:
    """
    Generates a conversational response from the Llama model.

    Parameters:
        message (str): User's input message.
        history (list): Past conversation history.

    Returns:
        str: Generated response from the Llama model.
    """
    query = format_message(message, history)
    response = ""
    for character in model(query, temperature = 0.3, 
                        repetition_penalty = 1.15, 
                        max_new_tokens=1000,stream=True):
        response += character
        yield response
"""
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def user(user_message, history):
        print(""+ str(history) + str([[user_message, None]] ))
        return "", history + [[user_message, None]]
    
    msg.submit(get_tinyllama_response, [msg,chatbot], [chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
"""
gr.ChatInterface(get_tinyllama_response).launch(inbrowser=True)
