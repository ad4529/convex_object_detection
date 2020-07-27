from multiprocessing import Pool
import numpy as np
import psutil
import os

def limit_cpu():
    p = psutil.Process(os.getpid())
    p.nice(19)
def mul(x):
    return np.linalg.norm(x)

if __name__=='__main__':
    a = [1,2,3]
    b = [4,5,6]
    c = [7,8,9]
    pool = Pool(None, limit_cpu)
    with pool as p:
        res = p.map(mul, [a,b,c])
    print(res)
