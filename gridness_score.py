import numpy as np
from matplotlib import pyplot as plt
from source import  *
from utils import draw_heatmap_2D

output_dir = './output/square_init110_loss9_lamda0.07_chosen/test'
num_interval = 40
block_size = 6
num_block = 16

weights_file = os.path.join(output_dir, 'weights_0.npy')
weights = np.load(weights_file)
weights = np.transpose(weights, axes=[2, 0, 1])
weights = np.clip(weights, a_min=0.0, a_max=weights.max())
score_list = np.zeros(shape=[len(weights)], dtype=np.float32)
ncol, nrow = block_size, num_block
plt.figure(figsize=(int(ncol * 2), int(nrow * 1.6)))

scale_list, orientation_list = [], []
for i in range(len(weights)):
    rate_map = weights[i]
    rate_map = (rate_map - rate_map.min()) / (rate_map.max() - rate_map.min())
    score, autocorr_ori, autocorr, scale, orientation = \
        gridnessScore(rateMap=rate_map, arenaDiam=1.0, h=1.0 / (num_interval-1), corr_cutRmin=0.3)
    # for p in range(len(peak)):
    #     autocorr_ori[peak[p, 0], peak[p, 1]] = autocorr.min() - 1000
    #     autocorr_ori[peak[p, 0]-1, peak[p, 1]] = autocorr.min() - 1000
    #     autocorr_ori[peak[p, 0]-1, peak[p, 1]-1] = autocorr.min() - 1000
    #     autocorr_ori[peak[p, 0], peak[p, 1]-1] = autocorr.min() - 1000
    #     autocorr_ori[peak[p, 0]+1, peak[p, 1]] = autocorr.min() - 1000
    #     autocorr_ori[peak[p, 0], peak[p, 1]+1] = autocorr.min() - 1000
    #     autocorr_ori[peak[p, 0]+1, peak[p, 1]+1] = autocorr.min() - 1000

    plt.subplot(nrow, ncol, i + 1)
    draw_heatmap_2D(autocorr_ori, vmax=autocorr.max())
    if score < 0.0:
        plt.title('%.2f' % score, color='red')
    else:
        plt.title('%.2f, %.2f, %.2f' % (score, scale, orientation), color='black')
        scale_list.append(scale)
        orientation_list.append(orientation)
    plt.subplots_adjust(wspace=1, hspace=0.5)

    score_list[i] = score
plt.savefig(os.path.join(output_dir, 'autoCorr.png'))
print(score_list)
print(np.sum(score_list > 0.0))

# plot histogram of scale
scale_list = np.asarray(scale_list)
plt.figure()
plt.hist(scale_list, bins=15)
plt.xlabel('Grid scale')
plt.ylabel('Frequency')
plt.savefig('scale_hist.png')

# plot histogram of orientation
orientation_list = np.asarray(orientation_list)
plt.figure()
plt.hist(orientation_list, bins=10)
plt.xlabel('Grid orientation')
plt.ylabel('Frequency')
plt.savefig('orientation_hist.png')

