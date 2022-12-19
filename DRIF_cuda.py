import numpy as np
from tauchen import tauchenhussey
import matplotlib.pyplot as plt
from numba import cuda, float32, void, int32
from compute_q import com_Q_recompile
from compute_DRIFExpectVc import com_ExpectVc_recompile
from compute_Vd import com_Vd_recompile
from compute_Vc import com_Vc_recompile
from compute_V0 import com_V0_recompile
import time
# get the start time
st = time.time()


Beta = 0.953 # discount factor (guess)
theta = 0.282
errorV = 100
# errord = 100
errorq = 100
tol = 10e-12
sigma = 2 #Ris aversion
eta = 0.025 # standard deviation of epli
rho = 0.945 #
ny = 51
mu = 0
[y21, yp21] = tauchenhussey(ny, mu, rho, eta, eta)
y = np.array(np.exp(y21).flatten(),dtype='float32')

Bmax = 0.35
Bmin = -0.35
nB  = 800
B = np.linspace(Bmin,Bmax,nB,dtype='float32')
indzero = np.searchsorted(B, 1e-10)

#B[int(nB/2)] = 0
r = 0.017 # Risk- Free interest rate
# guess of bond price schedule


# set up consumption
c_c = np.zeros(shape = (nB,ny))
c_d = np.zeros(shape = (ny),dtype='float32')
high, low = np.mean(y) * 1.05, np.mean(y) * .95
iy_high, iy_low = (np.searchsorted(y, x) for x in (high, low))
# compute the consumption c(B,y)
yhat = 0.969*np.mean(y)
for j in range(ny):
    c_d[j] = min(yhat, y[j])


## q
res_q = 1/(1+r)*np.ones(shape=(nB, ny),dtype='float32')
res_flat_q = np.array(res_q.flatten(), dtype='float32')
DIM_q_By = np.array([nB, ny], dtype='int32')

## V0
res_V0 = np.zeros(shape = (nB,ny),dtype='float32')
res_flat_V0 = np.array(res_V0.flatten(), dtype='float32')
DIM_V0_By = np.array([nB, ny], dtype='int32')
V0Old = np.zeros(shape = (nB,ny),dtype='float32')
V0New = np.zeros(shape = (nB,ny),dtype='float32')
## Vd

res_VdNew = np.zeros(shape = (ny), dtype= 'float32')
VdOld = np.zeros(shape = (ny) , dtype='float32')
res_flat_VdNew = np.array(res_VdNew.flatten(), dtype='float32')
DIM_Vd_y = np.array([ny], dtype='int32')

## Vc
# VcNew = np.zeros(shape = (nB,ny),dtype='float32')
res_Vc = np.zeros(shape =(nB,ny,nB),dtype='float32')
res_flat_Vc = np.array(res_Vc.flatten(), dtype='float32')
DIM_Valuec_ByB = np.array([nB, ny, nB], dtype='int32')

VcOld = np.zeros(shape =(nB,ny),dtype='float32')


##
#Bp = np.zeros(shape = (nB,ny))

iter  = 0
dp = np.zeros(shape = (nB,ny)) # default probability
DB = np.full((nB, ny), True)

res_ExpectVc = np.zeros(shape = (nB,ny),dtype='float32')
res_flat_ExpectVc = np.array(res_ExpectVc.flatten(), dtype='float32')
DIM_expectVc_By = np.array([nB, ny], dtype='int32')



## setting threadsperblock
threadsperblock = 256
threadsperblock_Vc = 256




##set block for Expectation of Vc, Vd, Vc, V0 q
blockspergrid_q = (res_flat_q.shape[0] + (threadsperblock - 1)) // threadsperblock
blockspergrid_ExpectVc = (res_flat_ExpectVc.shape[0] + (threadsperblock - 1)) // threadsperblock
blockspergrid_Vd = (res_flat_VdNew.shape[0] + (threadsperblock - 1)) // threadsperblock

blockspergrid_Vc = (res_flat_Vc.shape[0] + (threadsperblock_Vc - 1)) // threadsperblock
blockspergrid_V0 = (res_flat_V0.shape[0] + (threadsperblock - 1)) // threadsperblock



# compile kernel function of Expectation of Vc, Vd, V0
com_Q = com_Q_recompile(DIM_q_By,yp21,ny,r)
com_ExpectVc = com_ExpectVc_recompile(DIM_expectVc_By,yp21,ny)
com_Vd = com_Vd_recompile(theta,Beta,sigma,indzero, DIM_Vd_y,yp21,ny)
com_Vc = com_Vc_recompile(DIM_Valuec_ByB,y,B,sigma,Beta)
com_V0 = com_V0_recompile(DIM_V0_By)

## transfer data to cuda
d_c_d = cuda.to_device(c_d)


# d_res_expectVc_flat = cuda.to_device(res_flat_ExpectVc)
# d_res_VdNew_flat = cuda.to_device(res_flat_VdNew)
# d_res_Valuec_flat = cuda.to_device(res_flat_Valuec)
# d_res_V0New_flat = cuda.to_device(res_flat_V0)
# d_qOld = cuda.to_device(qOld)
# com_ExpectVc[blockspergrid_ExpectVc, threadsperblock](d_V0Old, d_res_expectVc_flat)


#com_Vd[blockspergrid_Vd, threadsperblock](d_VdOld,d_V0Old,d_c_d,d_res_VdNew_flat)
# expectVc =  d_res_expectVc_flat.copy_to_host().reshape(DIM_expectVc_By)
# d_expectVc = cuda.to_device(expectVc)
# com_Valuec[blockspergrid_Vc, threadsperblock](d_res_Valuec_flat, d_qOld, d_expectVc)
# Valuec_flat = d_res_Valuec_flat.copy_to_host()
# Valuec = Valuec_flat.reshape(DIM_Valuec_ByB)

# V = np.max(Valuec, axis=(2))

# idx_policy = np.argmax(Valuec.reshape((nB, ny, nB)), axis=2)
# idx_policy_k = np.unravel_index(idx_policy, (nB,ny))
# policy_B = B[idx_policy_k] # Bp



## Call expectation of Vc
#ExpectVc =  d_res_expectVc_flat.copy_to_host().reshape(DIM_expectVc_By)






# while errorq > tol:
st = time.time()
while errorV > tol:
    d_res_q_flat = cuda.to_device(res_flat_q)
    d_res_expectVc_flat = cuda.to_device(res_flat_ExpectVc)
    d_res_VdNew_flat = cuda.to_device(res_flat_VdNew)
    d_res_Vc_flat = cuda.to_device(res_flat_Vc)
    d_res_V0New_flat = cuda.to_device(res_flat_V0)
    d_V0Old = cuda.to_device(V0Old)
    d_VdOld = cuda.to_device(VdOld)
    d_VcOld = cuda.to_device(VcOld)
    iter = iter + 1
    print(iter)
    ## q
    com_Q[blockspergrid_q, threadsperblock](d_res_q_flat,d_VcOld,d_VdOld)
    q = d_res_q_flat.copy_to_host().reshape(DIM_q_By)
    d_q = cuda.to_device(q)
    ##compute expectation of Vc
    com_ExpectVc[blockspergrid_ExpectVc, threadsperblock](d_V0Old, d_res_expectVc_flat)
    expectVc = d_res_expectVc_flat.copy_to_host().reshape(DIM_expectVc_By)

    d_expectVc = cuda.to_device(expectVc)

    ## compute Vc
    com_Vc[blockspergrid_Vc, threadsperblock_Vc](d_res_Vc_flat, d_q, d_expectVc)

    Vc_result = d_res_Vc_flat.copy_to_host().reshape(DIM_Valuec_ByB)
    VcNew = np.max(Vc_result, axis=(2))

    idx_policy = np.argmax(Vc_result, axis=2)
    Bp = B[idx_policy]



    # compute Vd
    com_Vd[blockspergrid_Vd, threadsperblock](d_VdOld, d_V0Old, d_c_d, d_res_VdNew_flat)
    VdNew = d_res_VdNew_flat.copy_to_host().reshape(DIM_Vd_y)

    d_VcNew = cuda.to_device(VcNew)
    d_VdNew = cuda.to_device(VdNew)

    ## compute V0
    com_V0[blockspergrid_V0, threadsperblock](d_VcNew,d_VdNew,d_res_V0New_flat)
    V0New = d_res_V0New_flat.copy_to_host().reshape(DIM_V0_By)
    errorV = np.max(np.abs(V0New - V0Old)) + np.max(np.abs(VdNew - VdOld))
    V0Old = V0New.copy()
    VdOld = VdNew.copy()
    VcOld = VcNew.copy()
    print(f'error0 = {errorV}')

# # get the end time
et = time.time()
# # get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')



