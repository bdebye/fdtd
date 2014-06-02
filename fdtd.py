#!/usr/bin/python

from fdtd_W import *
from numpy import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.pylab as plb

#task.promote(1000e-11)
'''
brick = Brick(vector_3(1.0, 3.0, 0.0), vector_3(4.0, 4.0, 5.0))
med = Medium()
med.sig = 1e30

task.addMedium(brick, med)
'''
v = vector_3

Medium.eps0 = 8.854187817e-12
Medium.mu0 = 4e-7 * pi
Medium.sig0 = 0.0

Medium.silver = Medium(Medium.eps0, Medium.mu0, 6.17e7)
Medium.red_copper = Medium(Medium.eps0, Medium.mu0, 5.80e7)
Medium.brass = Medium(Medium.eps0, Medium.mu0, 1.57e7)
Medium.gold = Medium(Medium.eps0, Medium.mu0, 4.10e7)
Medium.concrete = Medium(8.16 * Medium.eps0, Medium.mu0, 0.001)
Medium.wood = Medium(2.1 * Medium.eps0, 1.00000043 * Medium.mu0, 0.0)

def show_profile(task, fn):
    data = zeros((task.Sx, task.Sy))
    for i in range(task.Sx):
        for j in range(task.Sy):
            data[i][j] = fn(task.Gp(i, j, task.Sz / 2))
    plb.matshow(data.transpose())
    plb.colorbar()
    plb.show()

def surf_profile(task):
    range_x = array(range(task.Sx)) * task.Ds
    range_y = array(range(task.Sy)) * task.Ds
    X, Y = meshgrid(range_x, range_y)
    Z = zeros((task.Sx, task.Sy))
    for i in range(task.Sx):
        for j in range(task.Sy):
            Z[i][j] = task.Ez(i, j, task.Sz / 2)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z.transpose(), rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth = 0, antialiased = False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()



def surf_profile_XZ(task):
    range_x = array(range(task.Sx)) * task.Ds
    range_z = array(range(task.Sz)) * task.Ds
    X, Y = meshgrid(range_z, range_x)
    Z = zeros((task.Sx, task.Sz))
    for i in range(task.Sx):
        for j in range(task.Sz):
            Z[i][j] = task.Ez(i, task.Sy / 2, j)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth = 0, antialiased = False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def show_signal(sig):
    X = map(lambda i: sig.time_pace * i, range(len(sig)))
    # Y = map(lambda i: sig[i], range(len(sig)))
    plt.plot(X, sig)
    plt.show()

if __name__ == '__main__':

    task = Fdtd()
    ct = Cartesian(vector_3(7, 5, 2.5), 0.05)
    task.setCoordinateSystem(ct)
    task.setTimePace(5e-11)
    
    task.addPointSource(vector_3(0.8, 1.6, 1.25), 0.25, 0.2)
    task.addReceivePoint(vector_3(5.6, 2.5, 1.8))
    
    '''
    task.addMedium(Brick(v(0.0, 1.0, 0.0), v(1.5, 4.0, 1.0)), Medium.wood)
    task.addMedium(Brick(v(2.2, 4.0, 0.0), v(5.0, 5.0, 2.0)), Medium.wood)
    task.addMedium(Brick(v(2.0, 0.0, 0.0), v(7.0, 1.0, 2.0)), Medium.wood)
    task.addMedium(Brick(v(6.0, 0.0, 0.0), v(7.0, 2.4, 2.0)), Medium.wood)
    task.setWall(0.2, Medium.concrete)
    '''
    #task.addMedium(Brick(v(2.0, 0.0, 0.0), v(5.0, 5.0, 2.5)), Medium.concrete)                            

    while True:
        N = int(input("ENTER:  "))
        #N = 3000
        if(N == -1):
            break
        task.promote(N * task.Dt)
        task.getReceiveSignal(0).save("signal.txt")
        #break
        show_profile(task, lambda c: c.E.z)
        show_signal(task.getReceiveSignal(0))


#task.promote(1e-11)
