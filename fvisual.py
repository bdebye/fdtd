from fdtd_W import *
from numpy import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.pylab as plb


def dependencies_for_myprogram():
    from scipy.sparse.csgraph import _validation
    import formlayout
    import mpl_toolkits.mplot3d

def plot_signal(sig):
    X = map(lambda i: sig.time_pace * i, range(len(sig)))
    plt.plot(X, sig)
    plt.show()
    
def surf_E_plane(task, pos, fc, fig = None):
    range_x = array(range(task.Sx)) * task.Ds
    range_y = array(range(task.Sy)) * task.Ds
    X, Y = meshgrid(range_x, range_y)
    Z = zeros((task.Sx, task.Sy))
    zn = int(round(pos / task.Ds))
    for i in range(task.Sx):
        for j in range(task.Sy):
            Z[i][j] = fc(task.Gp(i, j, zn))
    
    figure = fig
    if fig == None:
        figure = plt.figure()
    ax = figure.gca(projection='3d')
    ax.plot_surface(X, Y, Z.transpose(), rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth = 0, antialiased = False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    if fig == None:
        plt.show()
    
def surf_H_plane(task, pos, fc, fig = None):
    range_x = array(range(task.Sx)) * task.Ds
    range_z = array(range(task.Sz)) * task.Ds
    X, Y = meshgrid(range_z, range_x)
    Z = zeros((task.Sx, task.Sz))
    yn = int(round(pos / task.Ds))
    for i in range(task.Sx):
        for j in range(task.Sz):
            Z[i][j] = fc(task.Gp(i, yn, j))
    
    if fig == None:
        fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth = 0, antialiased = False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    if fig == None:
        plt.show()
