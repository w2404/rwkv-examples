import os, copy, types, gc
from tqdm import tqdm
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

os.environ["RWKV_JIT_ON"] = '1' # '1' or '0', please use torch 1.13+ and benchmark speed
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 150
FREE_GEN_LEN = 256

GEN_alpha_presence = 0.2 # Presence Penalty
GEN_alpha_frequency = 0.2 # Frequency Penalty
AVOID_REPEAT = '，：？！'

CHUNK_LEN = 256 # split input into chunks to save VRAM (shorter -> slower)


from rwkv.model import RWKV
from rwkv.utils import PIPELINE

import config
model = RWKV(model=config.m_out, strategy=config.strategy)
pipeline = PIPELINE(model, f"./20B_tokenizer.json")
END_OF_TEXT = 0
END_OF_LINE = 187

model_tokens = []
model_state = None

AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = pipeline.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd

def run_rnn(tokens, newline_adj = 0):
    global model_tokens, model_state
    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    #这里没必要开tqdm，提问阶段虽然这里会卡一下，但是后续字符生成会在这里重复跑，那里消耗更大
    for i in (range(0,len(tokens),CHUNK_LEN)):
        out, model_state = model.forward(tokens[i:i+CHUNK_LEN], model_state)
    out[END_OF_LINE] += newline_adj # adjust \n probability
    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    return out

def send1(m,newline_adj=0):
    print('send1',m.encode())
    return run_rnn(pipeline.encode(m),newline_adj=newline_adj)


def on_message(o):
    global model_tokens, model_state 
    if o['new']:
        model_state=None
        model_tokens=[]
        gc.collect()
        torch.cuda.empty_cache()
    out = send1(o['message'])
    if not o['reply']:return
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
            temperature=o['temperature'],
            top_p=o['top_p'],
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

