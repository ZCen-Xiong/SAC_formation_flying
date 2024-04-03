import math
import numpy as np
import scipy.linalg as linalg
import scipy.signal as signal
import  Dynam.CW_Prop as CW_prop
import  controlpy as cp


import numpy as np

mu = 398600.5e9
sma = 7108e3
nT = np.sqrt(mu / sma ** 3)
gamma = 0.95
dT = 2
Q11 = np.eye(3) * 1e3
Q22 = np.eye(3) * 2e2
Q = np.block([[Q11, np.zeros((3, 3))], [np.zeros((3, 3)), Q22]])
R = np.eye(3) * 1e11
N = np.zeros((2, 6))

# Discrete state
def discrete_state(dT, nT):
    Ac = np.zeros((6, 6))
    Bc = np.zeros((6, 3))
    Ac[0:3, 3:6] = np.eye(3)
    Ac[3, 5] = -2 * nT
    Ac[4, 1] = -nT ** 2
    Ac[5, 2] = 3 * nT ** 2
    Ac[5, 3] = 2 * nT
    
    Bc[3:6, 0:3] = np.eye(3)
    
    C = np.eye(6)
    D = np.zeros((6, 3))
    
    sys_d = signal.cont2discrete((Ac, Bc, C, D), dT)
    
    Ad = sys_d[0]
    Bd = sys_d[1]
    
    return Ad, Bd


Ad, Bd = discrete_state(dT, nT)

# LQR
P = linalg.solve_discrete_are(Ad, Bd, Q, R)
K = linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

t_period = np.arange(0, 2 * np.pi / nT, dT)
t_sc = np.arange(0, 2 * np.pi / nT, dT)
sc_len = len(t_sc)
X_trans = np.zeros((6, sc_len))

Xi_orbit = np.zeros((6, len(t_period)))
Xf_orbit = np.zeros((6, len(t_period)))

Xi_0 = np.array([100, 0, 0, 0, -200*nT, 0])
Xf_0 = np.array([-100, 100, 0, 0.0527, 0.2107, 0.1054])

Xi = CW_prop.Free(1 * np.pi / nT, 0, Xi_0, nT)
Xf = CW_prop.Free(0.5 * np.pi / nT, 0, Xf_0, nT)


for i, t_val in enumerate(t_period):
    Xi_orbit [:, i] = CW_prop.Free(t_val,0,Xi,nT);
    Xf_orbit [:, i] = CW_prop.Free(t_val,0,Xf,nT);
dV = 0

X_trans[:, 0] = Xi.flatten()
X_trans_ac = X_trans.copy()

for i in range(1, len(t_period)):
    dX_pre = X_trans[:, i - 1] - Xf_orbit[:, i - 1]
    u = -K @ dX_pre
    dX_new = Ad @ dX_pre + Bd @ u
    X_trans[:, i] = dX_new + Xf_orbit[:, i - 1]
    dV += np.linalg.norm(u * dT)
    X_trans_ac[:, i] = Ad @ X_trans_ac[:, i - 1] + Bd @ u

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(Xi_orbit[0, :], Xi_orbit[1, :], Xi_orbit[2, :], 'k')
ax.plot(Xi_orbit[0, 0], Xi_orbit[1, 0], Xi_orbit[2, 0], 'ko')
ax.plot(X_trans[0, :], X_trans[1, :], X_trans[2, :], 'b--')
ax.plot(Xf_orbit[0, :], Xf_orbit[1, :], Xf_orbit[2, :], 'r')
ax.plot(Xf_orbit[0, 0], Xf_orbit[1, 0], Xf_orbit[2, 0], 'rs')
ax.set_title('lqr dV=' + str(dV))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()



