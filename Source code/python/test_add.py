import numpy as np
import time
import matrix
N=1024
M=1024
x=np.random.randint(0,100,(N,M))
y=np.random.randint(0,100,(N,M))

cpu_starttime = time.perf_counter()
z=np.zeros((N,M))
for i in range(N):
    for j in range(M):
        z[i][j]=x[i][j]+y[i][j]
cpu_endtime = time.perf_counter()

gpu_starttime = time.perf_counter()
gpu_z=matrix.madd(x,y)
gpu_endtime = time.perf_counter()
if (z==gpu_z).all():
    print('Pass')
else:
    print('Fail')
print('Python Time:',(cpu_endtime-cpu_starttime)*1000,'ms')
print('GPU Time:',(gpu_endtime-gpu_starttime)*1000,'ms')