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
scale_list = np.zeros(shape=[len(weights)], dtype=np.float32)
orientation_list = np.zeros(shape=[len(weights)], dtype=np.float32)
ncol, nrow = block_size, num_block
plt.figure(figsize=(int(ncol * 2), int(nrow * 1.6)))

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
    plt.subplots_adjust(wspace=1, hspace=0.5)

    score_list[i] = score
    scale_list[i] = scale
    orientation_list[i] = orientation
plt.savefig(os.path.join(output_dir, 'autoCorr.png'))
print(score_list)
print(np.sum(score_list > 0.0))

# plot histogram of scale
plt.figure()
plt.hist(scale_list[score_list > 0.0], bins=15)
plt.xlabel('Grid scale')
plt.ylabel('Frequency')
plt.savefig('scale_hist.png')

# plot histogram of orientation
plt.figure()
plt.hist(orientation_list[score_list > 0.0], bins=10)
plt.xlabel('Grid orientation')
plt.ylabel('Frequency')
plt.savefig('orientation_hist.png')

# make the scatter plot
scale_list = np.reshape(scale_list, [num_block, block_size])
score_list = np.reshape(score_list, [num_block, block_size])
select_idx = np.where(np.sum(score_list < 0.0, axis=1) == 0)[0]
scale_avg = np.mean(scale_list[select_idx], axis=1)
alpha = np.array([3.9414601, 4.6574736, 11.874706, 16.970573, 17.36293, 35.70909, 39.14808, 39.710724, 44.097527,
                  56.250626, 57.031147, 61.706524, 73.04434, 85.69689, 87.490715, 94.69994])
alpha = alpha[select_idx]
plt.figure()
plt.scatter(1.0/np.sqrt(alpha), scale_avg)
plt.xlabel(r'Learned $1/\sqrt{\alpha_k}$')
plt.ylabel('Grid scale')
#plt.xlim((0.005, 0.06))
#plt.ylim((0.25, 0.8))
plt.savefig('scatter.png')

