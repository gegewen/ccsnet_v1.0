import sys
import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
plt.jet()

dnorm_temp = lambda a : (a ) * (180 - 30) + 30
dnorm_P = lambda a : (a) * (300 - 100) + 100

def get_dataset(name):
    hf_r = h5py.File(f'data_set/{name}.hdf5', 'r')
    variable = np.array(hf_r.get(name))
    hf_r.close()
    return variable

def make_SG(x, SG_model):
    sg = SG_model.predict(x)
    return sg[0,:,:,:,0]

def make_dP(x, dP_model):
    dp = dP_model.predict(x) * 300
    return dp[0,:,:,:,0]

def make_BXMF(x, dp, sg, BXMF_model):
    data_x_and_pred = np.concatenate([x, 
                                      sg[np.newaxis,...,np.newaxis], 
                                      dp[np.newaxis,...,np.newaxis]/600], axis=-1)
    bxmf = BXMF_model.predict(data_x_and_pred) * 0.038
    return bxmf[0,:,:,:,0]

def make_BYMF(x, bpr, sg, BYMF_model):
    temp = np.repeat(x[:,:,:,-4,:][:,:,:,np.newaxis,:], 24, axis=-2)
    x_input = np.concatenate([temp, (bpr - 100)/(565 - 100), sg], axis=-1)
    y = BYMF_model.predict(x_input)*0.1+0.9
    y[sg == 0] = 0
    return y

def make_BDENG(x, bpr, bymf, BDENG_model):
    temp = np.repeat(x[:,:,:,-4,:][:,:,:,np.newaxis,:], 24, axis=-2)
    bymf_x = np.copy(bymf)
    bymf_x[bymf_x == 0] = 0.9
    x_input = np.concatenate([temp, (bpr - 100)/(565 - 100), (bymf_x-0.9)/0.1], axis=-1)
    y = BDENG_model.predict(x_input) * 900 + 100
    y[bymf == 0] = 0
    return y

def make_BDENW(x, bpr, bxmf, BDENW_model):
    temp = np.repeat(x[:,:,:,-4,:][:,:,:,np.newaxis,:], 24, axis=-2)
    x_input = np.concatenate([temp, (bpr - 100)/(565 - 100), bxmf/0.038], axis=-1)
    y = BDENW_model.predict(x_input) * 400 + 700
    y[bxmf == 0] = 0
    return y

def make_BPR(x, dp):
    temperature = dnorm_temp(x[0, 0, 0, -4, 0])
    pressure =  dnorm_P(x[0, 0, 0, -3, 0])
    pinit = get_p_init(temperature, pressure)
    return dp + pinit[...]

def get_p_init(t, p):
    tt = np.array((float(t) - 30)/(170 - 30)).reshape((1, 1))
    pp = np.array((float(p) - 100) / (370 - 100)).reshape((1, 1))
    rho_w = PropsSI("D", "T", tt + 273.15, "P", pp * 1e5, "water") 

    p_same_rho = []
    for i in range(96):
        p_bottom = p + rho_w * 2.083330 * 9.8 / 100000
        p_same_rho.append((p_bottom + p) / 2)
        p = p_bottom
    p_init = np.array(p_same_rho)
    p_init = np.repeat(p_init[:, np.newaxis], 200, axis=1)
    return p_init[:,:,np.newaxis]

def predict_all(x, SG_model, dP_model, BXMF_model, BYMF_model, BDENG_model, BDENW_model):
    sg = make_SG(x, SG_model)
    dp = make_dP(x, dP_model)
    bpr = make_BPR(x, dp)
    bxmf = make_BXMF(x, bpr, sg, BXMF_model)
    bymf = make_BYMF(x, bpr[np.newaxis,:,:,:,np.newaxis], sg[np.newaxis,:,:,:,np.newaxis], BYMF_model)[0,...,0]
    bdenw = make_BDENW(x, bpr[np.newaxis,:,:,:,np.newaxis], bxmf[np.newaxis,:,:,:,np.newaxis], BDENG_model)[0,...,0]
    bdeng = make_BDENG(x, bpr[np.newaxis,:,:,:,np.newaxis], bymf[np.newaxis,:,:,:,np.newaxis], BDENW_model)[0,...,0]
    return sg, dp, bxmf, bymf, bdenw, bdeng

dz = 2.083330
dx = np.cumsum(3.5938*np.power(1.035012, range(200))) + 0.1
dx = np.insert(dx, 0, 0.1)

vol_pore = []
for i in range(200):
    vol_pore.append((dx[i+1]**2 - dx[i]**2) * math.pi * dz * 0.15)
vol_pore = np.repeat(np.array(vol_pore)[np.newaxis,:], 96, axis=0)

X, Y = np.meshgrid(dx, np.linspace(0,200,num = 96))

def all_plot(sg, dp, bxmf, sg_hat, dp_hat, bxmf_hat):
    plt.figure(figsize=(15,5))
    t = -1
    plt.subplot(3,3,1)
    plt.pcolor(X, Y, np.flipud(sg[:,:,t,0]))
    plt.colorbar(fraction=0.02)
    plt.xlim([0,2000])
    plt.clim([0,1])
    plt.title('true sg (-)')
    
    plt.subplot(3,3,4)
    plt.pcolor(X, Y, np.flipud(dp[:,:,t,0]))
    plt.colorbar(fraction=0.02)
    plt.xlim([0,2000])
    plt.clim([0,np.max(dp)])
    plt.title('true dP (bar)')

    plt.subplot(3,3,7)
    plt.pcolor(X, Y, np.flipud(bxmf[:,:,t,0]))
    plt.colorbar(fraction=0.02)
    plt.xlim([0,2000])
    plt.clim([0,np.max(bxmf)])
    plt.title('true bxmf (-)')

    plt.subplot(3,3,2)
    plt.pcolor(X, Y, np.flipud(sg_hat[:,:,t]))
    plt.colorbar(fraction=0.02)
    plt.xlim([0,2000])
    plt.clim([0,1])
    plt.title('pred sg (-)')
    
    plt.subplot(3,3,5)
    plt.pcolor(X, Y, np.flipud(dp_hat[:,:,t]))
    plt.colorbar(fraction=0.02)
    plt.xlim([0,2000])
    plt.clim([0,np.max(dp)])
    plt.title('pred dP (bar)')
    
    plt.subplot(3,3,8)
    plt.pcolor(X, Y, np.flipud(bxmf_hat[:,:,t]))
    plt.colorbar(fraction=0.02)
    plt.xlim([0,2000])
    plt.clim([0,np.max(bxmf)])
    plt.title('pred bxmf (-)')

    plt.subplot(3,3,3)
    plt.pcolor(X, Y, np.flipud(np.abs(sg_hat[:,:,t]-sg[:,:,t,0])))
    plt.colorbar(fraction=0.02)
    plt.xlim([0,2000])
    plt.clim([0,1])
    plt.title('err sg (-)')
    
    plt.subplot(3,3,6)
    plt.pcolor(X, Y, np.flipud(np.abs(dp_hat[:,:,t]-dp[:,:,t,0])))
    plt.colorbar(fraction=0.02)
    plt.xlim([0,2000])
    plt.clim([0,np.max(dp)])
    plt.title('err dP (bar)')
    
    plt.subplot(3,3,9)
    plt.pcolor(X, Y, np.flipud(np.abs(bxmf_hat[:,:,t]-bxmf[:,:,t,0])))
    plt.colorbar(fraction=0.02)
    plt.xlim([0,2000])
    plt.clim([0,np.max(bxmf)])
    plt.title('err bxmf (-)')

    plt.tight_layout() 
    plt.show()
    