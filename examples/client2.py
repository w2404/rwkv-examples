"""
问题是，难以确保rwkv可以像gpt一样正确回应autogpt
"""

import urllib.request
import json

GEN_TEMP = 1.1 # It could be a good idea to increase temp when top_p is low
GEN_TOP_P = 0.7 # Reduce top_p (to 0.5, 0.2, 0.1 etc.) for better Q&A accuracy (and less diversity)
def send(o):
    return urllib.request.urlopen('http://localhost:8008',json.dumps(o).encode())

o=json.load(open('/dev/shm/1'))
l=o['messages']

def tomsg(o):
    if o['role']=='system':
        return o['content']
    else:
        return o['role']+': '+o['content']+'\nEntrepreneur-GPT:'
r=send({'message':tomsg(l[0]),'new':True,'reply':False,'temperature':GEN_TEMP,'top_p':GEN_TOP_P,})
for i in l[1:-1]:
    r=send({'message':tomsg(l[0]),'new':False,'reply':False,'temperature':GEN_TEMP,'top_p':GEN_TOP_P,})

r=send({'message':tomsg(l[-1]),'new':False,'reply':True,'temperature':GEN_TEMP,'top_p':GEN_TOP_P,})

s=r.read(16)
while len(s)>0:
    print(s.decode(errors='ignore'),end='',flush=True)
    s=r.read(16)

