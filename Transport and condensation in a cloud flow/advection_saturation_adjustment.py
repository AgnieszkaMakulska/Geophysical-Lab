import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio
v_max=1.
L=H=1500
dx=dy=50
dt=0.1
I=int(L/dx)
J=int(L/dy)
Lv=2.5*10**6
Rv=461
kappa=0.286
cp=4184
epsilon=0.622
e0=611
T0=273
p0=100000
Rd=287
g=9.81

def pressure(z,rv,theta):
    return p0 *(1-kappa*g*(1+rv)*z/(theta*(Rd+Rv*rv)))**(1/kappa)
def exner(p):
    return (p/p0)**kappa
def e_s(theta,p):
    T=np.multiply(exner(p),theta)
    return e0*np.exp(-Lv/Rv *(1/T - 1/T0))
def r_vs(theta,p):
    return epsilon*e_s(theta,p)/(p-e_s(theta,p))
def F(delta,p,rv,theta):
   return r_vs(theta+ np.multiply(Lv/(cp*exner(p)),delta),p)+delta-rv
def Fp(delta,p,theta):
    theta_p=theta+np.multiply(Lv/(cp*exner(p)),delta)
    T_p=np.multiply(exner(p),theta_p)
    return epsilon*p/(p-e_s(theta_p,p))**2 *e0*Lv**2 /(Rv*cp*T_p**2) *np.exp(-Lv/Rv *(1/T-1/T0)) +1

def newton_method(delta0,p,rv,theta):
  delta=np.array(np.ones((I,J))*delta0)
  delta=np.reshape(delta,(1,I,J))
  for n in range(10**2):
    h=-F(delta[n],p,rv,theta)/Fp(delta[n],p,theta)
    if np.any(np.isnan(delta[n]+h)):
        break
    delta=np.concatenate((delta,np.reshape(delta[n]+h,(1,I,J))))
  return delta[-1]


def streamfunction(x,y):
    return (v_max*L/np.pi) *np.cos(2*np.pi*x/L-np.pi/2)*np.sin(np.pi*y/H+np.pi/2)
psi=np.ones((I+1,J+1))
psi=np.fromfunction(lambda i,j: streamfunction(i*dx,j*dy), (I+1,I+1))
psi=np.transpose(psi)
u= (np.roll(psi,-1,axis=1)-psi)/dy
v= -(np.roll(psi,-1,axis=0)-psi)/dx
u=u[:-1,:-1]
v=v[:-1,:-1]
#Boundary conditions:
v[0,:]=v[0,:]-v[0,:]
v[-1,:]=v[-1,:]-v[-1,:]
x=[i*dx for i in range(0,I)]
y=[i*dy for i in range(0,J)]
xv,yv=np.meshgrid(x,y)
plt.quiver(xv,yv,u,v)
plt.title('Velocity field')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

def flux(phi_L,phi_R,u):
    return (phi_L*(u+np.abs(u))/2 + phi_R*(u-np.abs(u))/2)*dt/dx
theta=np.ones((I,J))*298.
rv=np.ones((I,J))*0.017
rl=np.zeros((I,J))

T=200
for t in range(T):
    for i in range(1,I-1):
        for j in range(J):
            theta[i][j]= theta[i][j]-(flux(theta[i][j],theta[(i+1)%I][j],u[(i+1)%I][j])-flux(theta[(i-1)%I][j],theta[i][j],u[i][j]))
            theta[i][j]= theta[i][j]-(flux(theta[i][j],theta[i][(j+1)%J],v[i][(j+1)%J])-flux(theta[i][(j-1)%J],theta[i][j],v[i][j]))
            rv[i][j]= rv[i][j]-(flux(rv[i][j],rv[(i+1)%I][j],u[(i+1)%I][j])-flux(rv[(i-1)%I][j],rv[i][j],u[i][j]))
            rv[i][j]= rv[i][j]-(flux(rv[i][j],rv[i][(j+1)%J],v[i][(j+1)%J])-flux(rv[i][(j-1)%J],rv[i][j],v[i][j]))
            rl[i][j]= rl[i][j]-(flux(rl[i][j],rl[(i+1)%I][j],u[(i+1)%I][j])-flux(rl[(i-1)%I][j],rl[i][j],u[i][j]))
            rl[i][j]= rl[i][j]-(flux(rl[i][j],rl[i][(j+1)%J],v[i][(j+1)%J])-flux(rl[i][(j-1)%J],rl[i][j],v[i][j]))
    
    p=pressure(np.indices((30,30))[0]*dy,rv,theta)
    delta = (rv>=r_vs(theta,p)).astype(int)
    delta=np.multiply(delta,newton_method(0.,p,rv,theta))
    rv=rv-delta
    rl=rl+delta
    theta=theta+ Lv/(cp*exner(p))*delta
    
    if t%(T/20)==0:
        print(t/(T/20))
        plt.figure()
        #plt.imshow(rv,origin="lower",vmin=0., vmax=0.017)
        plt.imshow(rl*1000,origin="lower",vmin=0., vmax=7.)
        plt.colorbar(label='$r_l$ [g/kg]')
        plt.xticks(ticks=np.arange(0,30,6),labels=np.arange(0,1500,300))
        plt.yticks(ticks=np.arange(0,30,6),labels=np.arange(0,1500,300))
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Liquid water mixing ratio, '+'dx='+str(dx)+'m, dt='+str(dt)+'s, t='+ str(round(t*dt,1))+'s')
        plt.savefig(fname=str(int(t/(T/20))),dpi=200)
        plt.close()

frames = []
for t in range(20):
    image = imageio.v2.imread(f'./{t}.png')
    frames.append(image)
imageio.mimsave('./source.gif', # output gif
                frames,          # array of input frames
                fps = 2)         # optional: frames per second


plt.rcParams['font.size'] = 14
fig, ax = plt.subplots(1,3, figsize=(15,7))
ax[0].plot(np.mean(rv,axis=1)*10**3,np.arange(0,H,dy))
ax[1].plot(np.mean(rl,axis=1)*10**3,np.arange(0,H,dy))
ax[2].plot(np.mean(theta,axis=1),np.arange(0,H,dy))
ax[0].set_xlabel('$r_v$ [g/kg]')
ax[1].set_xlabel('$r_l$ [g/kg]')
ax[2].set_xlabel(r'$\theta$ [K]')
for i in range(3):
    ax[i].set_ylabel('z [m]')
ax[0].set_title('Water vapor mixing ratio')
ax[1].set_title('Liquid water mixing ratio')
ax[2].set_title('Potential temperature')
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.savefig(fname='profiles.png',dpi=300)
plt.show()