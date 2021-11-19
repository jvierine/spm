#!/usr/bin/env python
#
# Boris mover, model the motion of a charged particle in an electric field
#
# (c) 2015 Juha Vierinen
# 
import numpy as n
import scipy.constants as c
import matplotlib.pyplot as plt
import datetime
import jcoord
from numba import jit

@jit 
def E_field(t):
    return(n.array([0.0,0.0,0.0]))

# boris scheme, nit time steps
@jit
def move(x=n.array([0.0,0.0,0.0]),       # Initial position
         v=n.array([0.0,10.0,0.0]),      # Initial velocity
         E=E_field,       # Electric field (V/m)
         B=n.array([35000e-9,0,0]),      # Magnetic field (Tesla)
         dt=1.0/25e6,                    # time step \Delta t
         coll_rate=1e6,                  # collision frequency
         nit=10000):                     # time steps
    

    coll_prob=dt*coll_rate

    
    xt=n.zeros([nit,3])
    vt=n.zeros([nit,3])    
    tvec=n.arange(nit)*dt
    a=n.zeros(3,dtype=n.float32)
    qm = -c.elementary_charge/c.m_e    # q/m
    qp = dt*qm/2.0                    # q' = dt*q/2*m 
    for i in range(nit):              # From Wikipedia (Particle in Cell)
        h = qp*B                      # h=q'*B
        hs = n.sum(h**2.0)            # |h|^2
        s = 2.0*h/(1.0+hs)            # s          
        u = v + qp*E(i*dt)            # v^{n-1/2}
        up = u + n.cross((u + (n.cross(u,h))),s) # u' = u + (u + (u x h)) x s
        v = up + qp*E(i*dt)           # v^{n+1/2}
        x = x + dt*v

        # add collision, randomize new direction
        if n.random.rand(1) < coll_prob:
            v_unit = n.random.randn(3)
            v_unit=v_unit/n.linalg.norm(v_unit)
            v=n.linalg.norm(v)*v_unit
        
        xt[i,:]=x                     # store position
        vt[i,:]=v                     # store vel
    return(xt,vt)




#move(x=pos,       # Initial position
#     v=n.array([1000.0,1000.0,0.0]), # Initial velocity
#     E=n.array([0.0,0,0]),           # Electric field (V/m)
#     B=n.array([35000e-9,0,0]),      # Magnetic field (Tesla)
#     dt=1.0/25e6,                    # time step \Delta t
#     nit=10000):                     # time steps

