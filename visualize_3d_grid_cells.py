import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
from mayavi import mlab
import matplotlib.pyplot as plt

if __name__ == '__main__':

    weights_path = './training_result/learned_patterns/weights.npy'
    output_path = './training_result/learned_patterns/'
    num_row = 10
    num_col = 6

    weights = np.load(weights_path).transpose(3, 0, 1, 2)

    for i in range(len(weights)):
        volume = weights[i]
        [x_dim, y_dim, z_dim] = volume.shape

        fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))

        xslice = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(volume),
                                    plane_orientation='x_axes',
                                    slice_index=x_dim // 2,
                                    colormap='jet'
                                )
        yslice = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(volume),
                                    plane_orientation='y_axes',
                                    slice_index=y_dim // 2,
                                    colormap='jet'
                                )
        zslice = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(volume),
                                plane_orientation='z_axes',
                                slice_index=z_dim // 2,
                                colormap='jet'
                            )

        mlab.outline()


        # xslice = mlab.volume_slice(volume, colormap='jet', plane_orientation='x_axes', slice_index=x_dim // 2, figure=fig)
        # yslice = mlab.volume_slice(volume, colormap='jet', plane_orientation='y_axes', slice_index=y_dim // 2, figure=fig)
        # zslice = mlab.volume_slice(volume, colormap='jet', plane_orientation='z_axes', slice_index=z_dim // 2, figure=fig)

        # colorbar = mlab.colorbar(zslice, orientation='vertical')

        # colorbar.scalar_bar_representation.position=[0.9, 0.1]
        # colorbar.scalar_bar_representation.position2=[0.9, 0.9]

        mlab.view(150, 60, distance=130)
        #mlab.show()

        mlab.savefig(output_path + 'heatmap_' + str(i) + '.png', figure=fig)
        mlab.close()

