import urllib.request
q='小明:你好\n小红:你好\n小明:'
q='我:1+1=?\nAnswer:'

q='Bob: 1+1=?\n\nAlice: '
q='1+1=?'
r=urllib.request.urlopen('http://localhost:8008',q.encode())

s=r.read(16)
while len(s)>0:
    print(s.decode(errors='ignore'),end='',flush=True)
    s=r.read(16)

