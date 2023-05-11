import urllib.request
import json

def send(o):
    return urllib.request.urlopen('http://localhost:8008',json.dumps(o).encode())

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


PROMPT_FILE = f'./English-2.py' #{current_path}/prompt/default/{CHAT_LANG}-2.py'
user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)

#GEN_TEMP = 1.1 # It could be a good idea to increase temp when top_p is low
#GEN_TOP_P = 0.7 # Reduce top_p (to 0.5, 0.2, 0.1 etc.) for better Q&A accuracy (and less diversity)
r=send({'message':init_prompt,'new':True,'reply':False,'temperature':1.1,'top_p':0.7,})

while True:
    msg = input(':').replace('\\n','\n').strip()
    msg = msg.strip().replace('\r\n','\n').replace('\n\n','\n')
    new = f"{user}{interface} {msg}\n\n{bot}{interface}"
    r=send({'message':new,'new':False,'reply':True,'temperature':1.1,'top_p':0.7,})

    s=r.read(16)
    while len(s)>0:
        print(s.decode(errors='ignore'),end='',flush=True)
        s=r.read(16)

