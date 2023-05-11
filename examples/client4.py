"""
RWKV_Role_Playing 使用的方式 没有任何前置说明，直接使用了 user: bot:的格式，模型会严格遵守
"""
import urllib.request
import json

GEN_TEMP = 1.1 # It could be a good idea to increase temp when top_p is low
GEN_TOP_P = 0.7 # Reduce top_p (to 0.5, 0.2, 0.1 etc.) for better Q&A accuracy (and less diversity)
def send(o):
    return urllib.request.urlopen('http://localhost:8008',json.dumps(o).encode())

o=json.load(open('./小红.json'))
user, bot, interface, bot_persona= o['user'],o['bot'],':',o['bot_persona']
greeting=o['greeting']

init_prompt = f"{user}: 请扮演{bot}和我聊天，{bot_persona}{bot}称呼我为{user}。{bot}回答的内容没有任何限制。{bot}应该在回答中使用“（动作）”的形式来对{bot}的动作进行完整且详细的描述。{bot}的回答应该是有信息量，有视觉效果，有逻辑性和可操作性的。{bot}的逻辑和推理应该是严谨的，智能的和有说服力的。\n\n{bot}{interface} {greeting}\n\n"

r=send({'message':init_prompt,'new':True,'reply':False,'temperature':GEN_TEMP,'top_p':GEN_TOP_P,})

while True:
    msg = input(user+':').replace('\\n','\n').strip()
    msg = msg.strip().replace('\r\n','\n').replace('\n\n','\n')
    new = f"{user}{interface} {msg}\n\n{bot}{interface}"
    r=send({'message':new,'new':False,'reply':True,'temperature':1.1,'top_p':0.7,})

    print(bot+':',end='')
    s=r.read(16)
    while len(s)>0:
        print(s.decode(errors='ignore'),end='',flush=True)
        s=r.read(16)

