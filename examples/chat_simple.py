########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, copy, types, gc, sys
current_path = os.path.dirname(os.path.abspath(__file__))
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

os.environ["RWKV_JIT_ON"] = '1' # '1' or '0', please use torch 1.13+ and benchmark speed
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

CHAT_LANG = 'English' # English // Chinese // more to come

# Download RWKV models from https://huggingface.co/BlinkDL
# Use '/' in model path, instead of '\'
# Use convert_model.py to convert a model for a strategy, for faster loading & saves CPU RAM 

PROMPT_FILE = f'./English-2.py' #{current_path}/prompt/default/{CHAT_LANG}-2.py'

CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 150
FREE_GEN_LEN = 256

# For better chat & QA quality: reduce temp, reduce top-p, increase repetition penalties
# Explanation: https://platform.openai.com/docs/api-reference/parameter-details
GEN_TEMP = 1.1 # It could be a good idea to increase temp when top_p is low
GEN_TOP_P = 0.7 # Reduce top_p (to 0.5, 0.2, 0.1 etc.) for better Q&A accuracy (and less diversity)
GEN_alpha_presence = 0.2 # Presence Penalty
GEN_alpha_frequency = 0.2 # Frequency Penalty
AVOID_REPEAT = '，：？！'

CHUNK_LEN = 256 # split input into chunks to save VRAM (shorter -> slower)


########################################################################################################

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

def load_prompt(PROMPT_FILE):
    variables = {}
    with open(PROMPT_FILE, 'rb') as file:
        exec(compile(file.read(), PROMPT_FILE, 'exec'), variables)
    user, bot, interface, init_prompt = variables['user'], variables['bot'], variables['interface'], variables['init_prompt']
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
        init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n' + ('\n'.join(init_prompt)).strip() + '\n\n'
    return user, bot, interface, init_prompt

# Load Model

import config
model = RWKV(model=config.m_out, strategy=config.strategy)
pipeline = PIPELINE(model, f"{current_path}/20B_tokenizer.json")
END_OF_TEXT = 0
END_OF_LINE = 187
# pipeline = PIPELINE(model, "cl100k_base")
# END_OF_TEXT = 100257
# END_OF_LINE = 198

model_tokens = []
model_state = None

AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = pipeline.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd

########################################################################################################

def run_rnn(tokens, newline_adj = 0):
    global model_tokens, model_state

    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    # print(f'### model ###\n{tokens}\n[{pipeline.decode(model_tokens)}]')

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    out[END_OF_LINE] += newline_adj # adjust \n probability

    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    return out



def send1(m,newline_adj=0):
    print('send1',m.encode())
    return run_rnn(pipeline.encode(m),newline_adj=newline_adj)

########################################################################################################

# Run inference
print(f'\nRun prompt...')

user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)
out = send1(init_prompt) #run_rnn(pipeline.encode(init_prompt))
gc.collect()
torch.cuda.empty_cache()

srv_list = ['dummy_server']


def on_message(message):
    global model_tokens, model_state, user, bot, interface, init_prompt


    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P
    
    msg = message.replace('\\n','\n').strip()
    msg = msg.strip().replace('\r\n','\n').replace('\n\n','\n')
    new = f"{user}{interface} {msg}\n\n{bot}{interface}"
    # print(f'### add ###\n[{new}]')
    out = send1(new) # run_rnn(pipeline.encode(new), newline_adj=-999999999)

    begin = len(model_tokens)
    out_last = begin
    occurrence = {}
    for i in range(999):
        if i <= 0:
            newline_adj = -999999999
        elif i <= CHAT_LEN_SHORT:
            newline_adj = (i - CHAT_LEN_SHORT) / 10
        elif i <= CHAT_LEN_LONG:
            newline_adj = 0
        else:
            newline_adj = min(3, (i - CHAT_LEN_LONG) * 0.25) # MUST END THE GENERATION

        for n in occurrence:
            out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
        token = pipeline.sample_logits(
            out,
            temperature=x_temp,
            top_p=x_top_p,
        )
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        out = run_rnn([token], newline_adj=newline_adj)
        out[END_OF_TEXT] = -999999999  # disable <|endoftext|>

        xxx = pipeline.decode(model_tokens[out_last:])
        if '\ufffd' not in xxx: # avoid utf-8 display issues
            yield xxx
            out_last = begin + i + 1
        
        send_msg = pipeline.decode(model_tokens[begin:])
        if '\n\n' in send_msg:
            send_msg = send_msg.strip()
            break

