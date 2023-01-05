import numpy as np
import cu_matrix_add as cuadd
import cu_matrix_sub as cusub
import cu_matrix_mul as cumul
import cu_matrix_trans as cutrans

def madd(x,y):
    return cuadd.madd(x,y)

def msub(x,y):
    return cusub.msub(x,y)

def mmul(x,y):
    return cumul.mmul(x,y)

def mtrans(x):
    return cutrans.mtrans(x)