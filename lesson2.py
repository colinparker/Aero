import numpy as np
import matplotlib.pyplot as plt
from math import *

N = 200                           # Number of points in each direction
xStart,xEnd = -4.0,4.0            # x-direction boundaries
yStart,yEnd = -2.0,2.0            # y-direction boundaries
x = np.linspace(xStart,xEnd,N)    # x 1D-array
y = np.linspace(yStart,yEnd,N)    # y 1D-array
X,Y = np.meshgrid(x,y)            # generation of the mesh grid

#np.shape(X)

uinf=1.0
alphad=0.0
alpha=alphad*pi/180

ufreestream=uinf*cos(alpha)*np.ones((N,N),dtype=float)
vfreestream=uinf*sin(alpha)*np.ones((N,N),dtype=float)

psifreestream= uinf*cos(alpha)*Y-uinf*sin(alpha)*X

def getVelocity(sources,xs,ys,X,Y):
    u = sources/(2*pi)*(X-xs)/((X-xs)**2+(Y-ys)**2)
    v = sources/(2*pi)*(Y-ys)/((X-xs)**2+(Y-ys)**2)
    return u,v

def getStreamFunction(sources,xs,ys,X,Y):
    psi = sources /(2*pi)*np.arctan2((Y-ys),(X-xs))
    return psi
    
sources=5.0
xsource,ysource=-1.0,0.0

usource,vsource=getVelocity(sources,xsource,ysource,X,Y)

psisource= getStreamFunction(sources,xsource,ysource,X,Y)

u=ufreestream+usource
v=vfreestream+vsource
psi=psifreestream+psisource

##stagnation
xstag=xsource-((sources/(2*pi*uinf))*cos(alpha))
ystag=ysource-((sources/(2*pi*uinf))*sin(alpha))
psistag= sources /(2*pi)*np.arctan2((ystag-ysource),(xstag-xsource))
###PLOT
size=10
plt.figure(figsize=(size,(yEnd-yStart)/(xEnd-xStart)*size))
plt.grid(True)
plt.xlabel('x',fontsize=16)
plt.ylabel('y',fontsize=16)
plt.xlim(xStart,xEnd)
plt.ylim(yStart,yEnd)

plt.streamplot(X,Y,u,v,density=2.0,linewidth=1,arrowsize=1,arrowstyle='->')
plt.scatter(xsource,ysource,c='#CD2305',s=80,marker='o')
plt.scatter(xstag,ystag,c='#006633',s=80,marker='o')
if (alpha==0.0):
    plt.contour(X,Y,psi,\
            levels=[-psistag,+psistag],\
            colors='#CD2305',linewidths=2,linestyles='solid')
plt.show()

