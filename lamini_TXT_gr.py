from PIL import Image, ImageShow
from io import BytesIO
from rich import print
from tqdm.rich import trange, tqdm
from rich import console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
import warnings
warnings.filterwarnings(action='ignore')
import datetime
from rich.console import Console
console = Console(width=110)
from transformers import pipeline
import gradio as gr
import os


modelfile = "LaMini-Flan-T5-77M"
repetitionpenalty = 1.3
temperature = 0.3
contextlength=512
logfile = 'DebugLog.txt'
filename = 'TXTsummary.txt'

## Test  MBZUAI/LaMini-Flan-T5-77M
with console.status("Loading LaMini77M...",spinner="dots12"):
    model77 = pipeline('text2text-generation',model="model/")


# FUNCTION TO LOG ALL CHAT MESSAGES INTO chathistory.txt
def writehistory(text):
    with open('Lamini_TXTLOG-prompts.txt', 'a') as f:
        f.write(text)
        f.write('\n')
    f.close()

def SRS(fulltext, llmpipeline, chunks, overlap, filename):
  """
  SUS aka Summarize Rewrite Suggest
  Function that take a long string text and summarize it: returning also
  suggested QnA from the process
  The fulltext is split in Tokens according to chunks and overlap specified
  inputs:
  llmpipeline -> a transformers pipeline instance for the text generation
  fulltext - > string
  chunks, overlap - > integers
  filename -> string path where the logs will be saved - example 'AGOGO-AIsummary-history.txt'
  returns:
  final: string, the final stuffed summarization
  rew: a rewritten format of the summarization
  qna: a list of pairs in dict format {'question':item,'answer': res}
  """
  from langchain.document_loaders import TextLoader
  from langchain.text_splitter import TokenTextSplitter
  TOKENtext_splitter = TokenTextSplitter(chunk_size=chunks, chunk_overlap=overlap)
  sum_context = TOKENtext_splitter.split_text(fulltext) #create a list
  model77 = llmpipeline
  final = ''
  strt = datetime.datetime.now()
  for i in trange(0,len(sum_context)):
    text = sum_context[i]
    template_bullets = f'''ARTICLE: {text}

    What is a one-paragraph summary of the above article?

    '''
    res = model77(template_bullets, temperature=0.3, repetition_penalty=1.3, max_length=400, do_sample=True)[0]['generated_text']
    final = final + ' '+ res+'\n'

  ## REWRITE FUNCTION
  template_final = f'''Rewrite the following text in and easy to understand tone: {final}
  '''
  rew = model77(template_final, temperature=0.3, repetition_penalty=1.3, max_length=400, do_sample=True)[0]['generated_text']
  elaps = datetime.datetime.now() - strt
  console.print(Markdown("# SUMMARY and REWRITE"))
  console.print(Markdown(final))
  console.print("[green2 bold]---")
  console.print(Markdown(rew))
  console.print(f"[red1 bold]Full RAG PREPROCESSING Completed in {elaps}")
  console.print(Markdown("---"))
  logger = "# SUMMARY and REWRITE\n---\n#SUMMAR\n" + final + "\n#REWRITTEN\n"+ rew + "\n---\n\n"
  writehistory(logger)

  # Generate Suggested questions from the text
  # Then Reply to the questions
  console.print(f"[green2 bold]Generating Qna...")
  finalqna = ''
  strt = datetime.datetime.now()
  for i in trange(0,len(sum_context)):
    text = sum_context[i]
    template_final = f'''{text}.\nAsk few question about this article.
  '''
    res = model77(template_final, temperature=0.3, repetition_penalty=1.3, max_length=400, do_sample=True)[0]['generated_text']
    finalqna = finalqna + '\n '+ res

  delt = datetime.datetime.now()-strt
  console.print(Markdown("---"))
  console.print(f"[red1 bold]Questions generated in {delt}")
  lst = finalqna.split('\n')
  final_lst = []
  for items in lst:
    if items == '':
      pass
    else:
      final_lst.append(items)

  qna = []
  for item in final_lst:
    question = item

    template_qna = f'''Read this and answer the question. If the question is unanswerable, ""say \"unanswerable\".\n\n{final}\n\n{question}
    '''

    start = datetime.datetime.now()
    res = model77(template_qna, temperature=0.3, repetition_penalty=1.3, max_length=400, do_sample=True)[0]['generated_text']
    elaps = datetime.datetime.now() - start
    """
    console.print(f"[bold deep_sky_blue1]{question}")
    console.print(Markdown(res))
    console.print(f"[red1 bold]Qna Completed in {elaps}")
    console.print(Markdown("---"))
    """
    qna.append({'question':item,
                'answer': res})
    logger = "QUESTION: " + question + "\nANSWER: "+ res + "\n---\n\n"
    writehistory(logger)

  return final, rew, qna

#Funtion to club the ones above

def start_SRS(fulltext, filename):
  console.print("Starting TXT Summarization...")
  global SUMfinal
  global REWSum
  global QNASet
  SUMfinal,REWSum,QNASet = SRS(fulltext, model77, 450, 10, filename)
  console.print("Process Completed!")
  return SUMfinal

def printQNA():
  head = """### Generated Qna<br><br>"""
  qnaset = " "
  for item in QNASet:
    temp = f"""
> **Question: {item['question']}**<br>
> *Answer: {item['answer']}*<br>
"""
    qnaset = qnaset + temp
  final = head + qnaset  
  return gr.Markdown(value=final)

def start_search(b):
  global fulltext
  with open(b.name, encoding="utf8") as f:
    fulltext = f.read()
  f.close()
  console.print("[bold deep_sky_blue1]text has been captured")
  console.print(Markdown("---"))
  console.print("Saved in variable `fulltext`")
  statmessage = "Operation Successful\nClick on Start Summarization ↗️" 
  return statmessage, fulltext

def generate(instruction):
  prompt = instruction
  start = datetime.datetime.now()
  with console.status("AI is thinking...",spinner="dots12"):
    output = model77(prompt, temperature=0.3, repetition_penalty=1.3, max_length=400, do_sample=True)[0]['generated_text']
  delta = datetime.datetime.now()-start  
  console.print(f"[green1 italic]:two_o’clock: generated in {delta}")
  console.print(f"[bold bright_yellow]🦙 LaMini-Flan-77M: {output}")
  return output, delta

# MAIN GRADIO INTERFACE
with gr.Blocks(theme=gr.themes.Base()) as demo:   #theme=gr.themes.Glass()
    #TITLE SECTION
    with gr.Row(variant='compact'):
            gr.HTML("<center>"
            + "<h1>Summarize your TXT files</h1>"
            + "<h2>🦙 Using LaMini-Flan-77M</h2>"
            + "<p>Create a brief Summary and Initial Insights</p></center>")  
    # MODEL PARAMETERS INFO SECTION
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            - **Prompt Template**: T5
            - **Repetition Penalty**: {repetitionpenalty}\n
            """)                            
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            - **Context Lenght**: {contextlength} tokens
            - **Log File**: {logfile}
            """)        
        with gr.Column(scale=2):
            gr.Markdown(
            f"""
            - **Model**: {modelfile}
            - **LLM Engine**: HuggingFace Transformers
            """)
    # PLAYGROUND INTERFACE SECTION
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            ### Instructions""")
            #temp = gr.Slider(label="Temperature",minimum=0.0, maximum=1.0, step=0.01, value=0.3)
            #max_len = gr.Slider(label="Maximum output lenght", minimum=10,maximum=512,step=2, value=400)
            gr.Markdown(
            """
            1. Pick your TXT file
            2. Click on load the Text
            """)
            urltext = gr.File(label="Your Text file", file_types=['.txt'], height=100,file_count="single")
            rawtext = gr.Textbox(label="extracted text", lines=1,visible=False )
            btn = gr.Button(value="1. 💾 Load the text", variant='primary')
            StatusUp = gr.Textbox(label="Loading status:", lines=1)

            
            btn.click(start_search, inputs=[urltext], outputs=[StatusUp,rawtext])

        with gr.Column(scale=4):
            btn3 = gr.Button(value="2.🦙 Start Summarization", variant='primary')
            txt = gr.Textbox(label="Summary", lines=12,show_copy_button=True)
            btn3.click(start_SRS, inputs=[rawtext], outputs=[txt])
            #
            btn2 = gr.Button(value="3.🧙‍♂️ Generate QnA", variant='primary')
            #ttitle = gr.Textbox(value='', label="Question&Answers:", 
            #                   lines=12,interactive=True,show_copy_button=True)
            with gr.Accordion("Question&Answers"):
              ttile = gr.Markdown(label="")          
            btn2.click(printQNA, inputs=[], outputs=[ttile])
            # NEXT is to put ACCORDIONS that changes according to the number of Qna
            # ref here https://www.gradio.app/guides/controlling-layout#variable-number-of-outputs

if __name__ == "__main__":
    demo.launch(inbrowser=True)
