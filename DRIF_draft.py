import numpy as np
from tauchen import tauchenhussey
import matplotlib.pyplot as plt
import os
import time


file_path = r'C:\Users\tangw\OneDrive\Desktop'
Beta = 0.953 # discount factor (guess)
theta = 0.282
dist = 100
errorq = 100
tol = 10e-12
sigma = 2 #Ris aversion
eta = 0.025 # standard deviation of epli
rho = 0.945 #
ny = 51
mu = 0
[y21, yp21] = tauchenhussey(ny, mu, rho, eta, eta)
y = np.exp(y21).flatten()
Bmax = 0.35
Bmin = -0.35
nB  = 200
B = np.linspace(Bmin,Bmax,nB)
res = np.searchsorted(B, 1e-10)
r = 0.017 # Risk- Free interest rate
# guess of bond price schedule
q = 1/(1+r)*np.ones(shape=(nB, ny))
# set up consumption
c_c = np.zeros(shape = (nB,ny))
c_d = np.zeros(shape = (ny))
# compute the consumption c(B,y)
yhat = 0.969*np.mean(y)
high, low = np.mean(y) * 1.05, np.mean(y) * .95
iy_high, iy_low = (np.searchsorted(y, x) for x in (high, low))
for j in range(ny):
    c_d[j] = min(yhat, y[j])


## solve value function iteration  when the government choose to repay
v0Old = np.zeros(shape = (nB,ny))#np.load(file_path + os.sep + 'dV0.npy')#np.zeros(shape = (nB,ny))#

Value_c = np.zeros(shape =(nB))
vdNew   = np.zeros(shape = (ny))
v0New = np.zeros(shape = (nB,ny))
vdOld = np.zeros(shape = (ny))#np.load(file_path + os.sep + 'dVd.npy')#np.zeros(shape = (ny))#
Bp    = np.zeros(shape = (nB,ny))
vcNew = np.zeros(shape = (nB,ny))
vcOld = np.zeros(shape = (nB,ny))#np.load(file_path + os.sep + 'dVc.npy')#np.zeros(shape = (nB,ny))#
expect_c = np.zeros(shape = (nB,ny))
iter  = 0
dp = np.zeros(shape = (nB,ny)) # default probability
DB = np.full((nB, ny), True)

# get the start time
st = time.time()
while dist > tol:
    iter = iter + 1
    print(iter)
    for i in range(nB):
        for j in range(ny):
            if vcOld[i, j] < vdOld[j]:
                DB[i, j] = True
            else:
                DB[i, j] = False
    ## update bond price
    for i in range(nB):
        for j in range(ny):
            dp[i, j] = sum(yp21[j, DB[i, :]])
            q[i, j] = (1 - dp[i, j]) / (1 + r)

    ##compute expectation
    for iBp in range(nB):
        for iy in range(ny):
            expect = 0
            for iyp in range(ny):
                expect = expect + v0Old[iBp, iyp] * yp21[iy, iyp]
            expect_c[iBp, iy] = expect
    for iB in range(nB):
        for iy in range(ny):
            #compute vNew
            for iBp in range(nB):
                Value_c[iBp] = (((y[iy]-q[iBp,iy]*B[iBp]+B[iB]) ** (1 - sigma)) / (1 - sigma)) + Beta * expect_c[iBp, iy]
            vcNew[iB, iy] = np.max(Value_c)
            idx_Bp = np.unravel_index(np.argmax(Value_c), Value_c.shape)
            Bp[iB][iy] = B[idx_Bp]
    # compute value function of Vd
    for iyd in range(ny):
        # expectation of Vd
        expect_d = 0
        for iypd in range(ny):
            expect_d = expect_d + (theta * v0Old[res, iypd] + (1 - theta) * vdOld[iypd]) * yp21[iyd, iypd]
        vdNew[iyd] = ((c_d[iyd]**(1-sigma))/(1-sigma)) + Beta * expect_d

    for i in range(nB):
        for j in range(ny):
            v0New[i,j] = max(vcNew[i,j], vdNew[j])

    dist = np.max(np.abs(v0New - v0Old)) + np.max(np.abs(vdNew - vdOld))
    v0Old = v0New.copy()
    vdOld = vdNew.copy()
    vcOld = vcNew.copy()
    print(f'error = {dist}')


# # get the end time
et = time.time()
# # get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')











