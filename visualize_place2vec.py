import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, axis3d, proj3d
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch


def colormap_2d(length_x=40, length_y=40):

    x = np.linspace(-3., 3., length_x)
    y = np.linspace(-3., 3., length_y)
    pos1, pos2 = np.meshgrid(x, y)
    X = pos1.ravel()
    Y = pos2.ravel()

    color2 = np.max([X+2.5, np.zeros(length_x*length_y)], axis=0)+np.max([Y+2.5, np.zeros(length_x*length_y)],axis=0)
    color3 = -np.min([Y-2.5, np.zeros(length_x*length_y)], axis=0)
    color1 = -np.min([X-2.5, np.zeros(length_x*length_y)], axis=0)

    color1 = (color1-color1.min())/(color1.max()-color1.min())
    color2 = (color2-color2.min())/(color2.max()-color2.min())
    color3 = (color3-color3.min())/(color3.max()-color3.min())

    color = np.array([color1, color2, color3]).T

    return color

def draw_points(pts, clr, cmap, ax=None, sz=20):

    if ax is None:
        fig = plt.figure()
        ax = axes3d.Axes3D(fig)
        ax.view_init(15, -64)
    else:
        ax.cla()

    ax.set_alpha(255)
    ax.set_aspect('equal')
    min_lim = pts.min()
    max_lim = pts.max()
    ax.set_xlim3d(min_lim,max_lim)
    ax.set_ylim3d(min_lim,max_lim)
    ax.set_zlim3d(min_lim,max_lim)

    if cmap is None and clr is not None:
        assert(np.all(clr.shape==pts.shape))
        sct=ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=clr,
            zdir='y',
            s=sz,
            edgecolors=(0.5, 0.5, 0.5)
        )

    else:
        if clr is None:
            M = ax.get_proj()
            _,clr,_ = proj3d.proj_transform(pts[:,0], pts[:,1], pts[:,2], M)
        clr = (clr-clr.min())/(clr.max()-clr.min()) #normalization
        sct=ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=clr,
            zdir='y',
            s=sz,
            cmap=cmap,
            edgecolors=(0.5, 0.5, 0.5)
        )

    ax.set_facecolor("white")
    return ax, sct


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


if __name__ == '__main__':

    data_path = './weights/weights_s0.7.npy'
    #data_path =  './weights_96dim.npy'
    length_of_size = 40
    num_points = length_of_size ** 2


    a = np.arange(0, num_points)

    clr = colormap_2d(length_of_size, length_of_size)
    x1 = np.arange(0, length_of_size)
    y1 = np.arange(0, length_of_size)
    pos1_1, pos2_1 = np.meshgrid(x1, y1)
    X_1 = pos1_1.ravel()
    Y_1 = pos2_1.ravel()
    plt.figure()
    plt.scatter(X_1[a], length_of_size-Y_1[a], c=clr[a], s=50)

    data = np.load(data_path)

    X = data.reshape(num_points, data.shape[2])


    if not data.shape[2] == 3:
        # PCA dimension reduction
        pca = decomposition.PCA(n_components=3)
        pca.fit(X)
        X = pca.transform(X)


    vect_length = np.sum(np.square(X), axis=1)


    ax, _ = draw_points(X[a], clr=clr[a], cmap=None, sz=30)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.axis('off')

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray")


    ax.add_artist(Arrow3D([0, 1], [0, 0], 
                [0, 0], mutation_scale=20, 
                lw=1, arrowstyle="-|>", color="black"))
    
    ax.text(1, 0, 0, 'x', color='black')


    ax.add_artist(Arrow3D([0, 0], [0, 1], 
                [0, 0], mutation_scale=20, 
                lw=1, arrowstyle="-|>", color="black"))
    ax.text(0, 1, 0, 'y', color='black')

    ax.add_artist(Arrow3D([0, 0], [0, 0], 
                [0, 1], mutation_scale=20, 
                lw=1, arrowstyle="-|>", color="black"))
    ax.text(0, 0, 1, 'z', color='black')

    ax.text(0, 0, 0, "(0, 0, 0)", color='black')

    plt.show()

