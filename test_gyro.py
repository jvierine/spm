import boris_mover as bm
import matplotlib.pyplot as plt
import numpy as n
import scipy.constants as c
import matplotlib.animation as animation
from numba import jit

@jit
def E_field(t):
    return(n.array([0,n.cos(2.0*n.pi*f*t),n.sin(2.0*n.pi*f*t)]))

n_it=100000
dt=1.0/100e6
tv=n.arange(n_it)*dt
B=50000e-9


gyro_freq=c.elementary_charge*B/c.electron_mass/2.0/n.pi

freqs=n.array([100e3])+gyro_freq

for f in freqs:
    # O-mode
 #   def E_field(t):
  #      return(n.array([n.cos(2.0*n.pi*f*t),n.sin(2.0*n.pi*f*t),0]))
    # X-mode
    def E_field(t):
        return(n.array([n.cos(-2.0*n.pi*f*t),n.sin(-2.0*n.pi*f*t),0]))
    
    x,v=bm.move(v=n.array([100000,0,0]),B=n.array([0.0,0.0,-B]),E=E_field,dt=dt,nit=n_it,coll_rate=0.0)#10*gyro_freq)

    ke = 0.5*(v[:,0]**2.0+v[:,1]**2.0+v[:,2]**2.0)*c.electron_mass/c.eV

 #   plt.plot(ke)
#    plt.show()
    fig,ax=plt.subplots()
    #    ax.set_title("B up")
    ax.set_title("B down $f=%1.2f$ MHz $f_g=%1.2f$ MHz mean kinetic energy %1.2g [%1.2g,%1.2g]"%(f/1e6,gyro_freq/1e6,n.mean(ke),n.min(ke),n.max(ke)))

    cpx=n.median(x[:,0])
    cpy=n.median(x[:,1])
    cpr=n.median(n.sqrt((x[:,0]-cpx)**2.0+(x[:,1]-cpy)**2.0))
    ax.text(n.median(x[:,0]),n.median(x[:,1]),"$\otimes$",size=20)
    li=ax.plot(x[:,0],x[:,1])
    sc=ax.scatter([x[0,0]], [x[0,1]],color="red")
    
    eli,=ax.plot([cpx,cpr*E_field(0*dt)[0]+cpx],[cpy,cpr*E_field(0*dt)[1]+cpy])
    

    AD=1
    def animate(i):
        print(i)
        sc.set_offsets(n.array([x[i*AD%x.shape[0],0],x[i*AD%x.shape[0],1]]))
        eli.set_xdata([cpx,cpr*E_field(i*AD*dt)[0]+cpx])
        eli.set_ydata([cpy,cpr*E_field(i*AD*dt)[1]+cpy])
        return(sc,eli,)
        
    ani = animation.FuncAnimation(
        fig, animate, interval=50, blit=True)
    plt.show()
    plt.xlabel("Y")
    plt.ylabel("Z")

    plt.show()
    
#    fig = plt.figure()
#    ax = fig.add_subplot(projection='3d')
    
#    ax.plot(tv,x[:,1],x[:,2])
    
 #   plt.show()

