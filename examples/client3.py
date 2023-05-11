"""
以指令的方式测试执行效果
这是+i的实际执行方式
"""
import urllib.request
import json

GEN_TEMP = 1.1 # It could be a good idea to increase temp when top_p is low
GEN_TOP_P = 0.7 # Reduce top_p (to 0.5, 0.2, 0.1 etc.) for better Q&A accuracy (and less diversity)
def send(o):
    return urllib.request.urlopen('http://localhost:8008',json.dumps(o).encode())

msg='write a python helloworld script' #成功 14B
msg='写一个python脚本抓取当前股票价格' #成功 14B

#这里读取一份原本auto要发送给gpt的参数
o=json.load(open('/dev/shm/1'))
msg=''
for i in o['messages']:
    #基本上我们要丢弃role，原因在于，autogpt中的role毕竟也是用作下指令的，所以role没有必要
    msg+=i['content']+'\n'
#执行结果确实是json格式，但是最初的goal是查询介绍一个东西，而返回的json中说的，却是执行python文件，删除文件之类的，其中的参数甚至是照搬了指令样例。
#可以说基本上是乱写的


#下面这个格式大概是rwkv被训练的？
new = f'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{msg}

# Response:
'''


r=send({'message':new,'new':True,'reply':True,'temperature':GEN_TEMP,'top_p':GEN_TOP_P,})
s=r.read(16)
while len(s)>0:
    print(s.decode(errors='ignore'),end='',flush=True)
    s=r.read(16)

