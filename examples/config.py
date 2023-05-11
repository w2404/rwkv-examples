import os

#14B在16G+24G的内存下转换失败，16G+38G转换成功
#16G运行14B是不够的，需要缓存。 
#一个问题是，当服务器启动后，我们能够看见的缓存和内存占用总数并不多，但是如果执行swapoff就会导致溢出。
#可能的原因是，1缓存被压缩了，2一些buffer没有被显示出来，实际占用并不少，有可能是vram的对等


#这个模型似乎有问题，大概迭代很少？至少不能rpg
m_in='/mnt/raid/6/@data/ai/rwkv/models/RWKV-4-ChnTest4-14B-20230430-ctx4096.pth'
m_in='/mnt/raid/6/@data/ai/rwkv/models/RWKV-4-Raven-7B-v11-Eng49-Chn49-Jpn1-Other1-20230430-ctx8192.pth'

#这个cpu版几乎不可用，占用大量内存，在我们没有内存的情况下，时间都耗费在缓存上了
#strategy='cuda fp16i8 *20 -> cpu fp32'
strategy,m_in='cuda fp16i8 *18+','/mnt/raid/6/@data/ai/rwkv/models/RWKV-4-Raven-14B-v11x-Eng99-Other1-20230501-ctx8192.pth'#峰值7.6G，但是在100pets提问中溢出了
strategy,m_in='cuda fp16i8','/mnt/raid/6/@data/ai/rwkv/models/RWKV-4-Raven-3B-v10x-Eng49-Chn50-Other1-20230423-ctx4096.pth'
strategy,m_in='cuda fp16i8 *15+','/mnt/raid/6/@data/ai/rwkv/models/RWKV-4-Raven-14B-v11x-Eng99-Other1-20230501-ctx8192.pth'#
strategy,m_in='cuda fp16i8 *30+','/mnt/raid/6/@data/ai/rwkv/models/RWKV-4-Raven-7B-v10x-Eng49-Chn50-Other1-20230423-ctx4096.pth'#峰值7.8G

p0,n=os.path.split(m_in)

m_out=p0+'/comp/'+strategy.replace(' ','_').replace('*','star').replace('+','plus').replace('>','-') +'-'+n
m_out='/home/rwkv/models/'+strategy.replace(' ','_').replace('*','star').replace('+','plus').replace('>','-') +'-'+n

print(m_out)
