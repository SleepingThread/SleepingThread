"""
#X.shape = (n_pts,)
#Y.shape = (n_pts,)
#Z.shape = (n_pts,)
#triangles.shape = (n_tri,3)
ax.plot_trisurf(X,Y,Z,triangles=triangles)
"""

def test3D():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x = [1,2,3]
    y = [1,2,3]
    z = [1,2,3]
    ax.plot(x,y,z)
    plt.show()


