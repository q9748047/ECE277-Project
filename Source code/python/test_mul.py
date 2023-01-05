import numpy as np
import time
import matrix
N=256
M=256
s=128
x=np.random.randint(0,100,(N,s))
y=np.random.randint(0,100,(s,M))

cpu_starttime = time.perf_counter()
z=np.zeros((N,M))
for i in range(N):
    for j in range(M):
        tmp=0
        for p in range(s):
            tmp+=x[i][p]*y[p][j]
        z[i][j]=tmp
cpu_endtime = time.perf_counter()

gpu_starttime = time.perf_counter()
gpu_z=matrix.mmul(x,y)
gpu_endtime = time.perf_counter()
if (z==gpu_z).all():
    print('Pass')
else:
    print('Fail')
print('Python Time:',(cpu_endtime-cpu_starttime)*1000,'ms')
print('GPU Time:',(gpu_endtime-gpu_starttime)*1000,'ms')