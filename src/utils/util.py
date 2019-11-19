
import numpy as np
import time

def condensing_matrix(size_z, size_a, in_channel):
    assert size_z % 2 == 1 and size_a % 2== 1, \
        'size_z and size_a should be odd number'

    half_filter_dim = (size_z * size_a)//2

    # moving neigboring pixels to channel dimension
    nbr2ch_mat = np.zeros(
        (size_z, size_a, in_channel, size_z*size_a*in_channel),
        dtype=np.float32
    )

    for z in range(size_z):
        for a in range(size_a):
            for ch in range(in_channel):
                nbr2ch_mat[z, a, ch, z*(size_a*in_channel) + a*in_channel + ch] = 1

    # exclude the channel index corresponding to the center position
    nbr2ch_mat = np.concatenate(
        [nbr2ch_mat[:, :, :, :in_channel*half_filter_dim],
         nbr2ch_mat[:, :, :, in_channel*(half_filter_dim+1):]],
        axis=3
    )
    assert nbr2ch_mat.shape == \
           (size_z, size_a, in_channel, (size_a*size_z-1)*in_channel),\
    'error with the shape of nbr2ch_mat after removing center position'

    return nbr2ch_mat

def angular_filter_kernel(size_z, size_a, in_channel, theta_sqs):
    """Compute a gaussian kernel.
    Args:
      size_z: size on the z dimension.
      size_a: size on the a dimension.
      in_channel: input (and output) channel size
      theta_sqs: an array with length == in_channel. Contains variance for
          gaussian kernel for each channel.
    Returns:
      kernel: ND array of size [size_z, size_a, in_channel, in_channel], which is
          just guassian kernel parameters for each channel.
    """
    assert size_z % 2 == 1 and size_a % 2 ==1, \
        'size_z and size_a should be odd number'
    assert len(theta_sqs) == in_channel, \
    'length of theta_sqs and in_channel does no match'

    #guassian kernel
    kernel = np.zeros((size_z, size_a, in_channel, in_channel),dtype=np.float32)
    for k in range(in_channel):
        kernel_2d = np.zeros((size_z, size_a),dtype=np.float32)
        for i in range(size_z):
            for j in range(size_a):
                diff = np.sum(
                    (np.array([i-size_z//2, j-size_a//2]))**2)
                kernel_2d[i, j] = np.exp(-diff/2/theta_sqs[k])

        # exklude the center position
        kernel_2d[size_z//2, size_a//2] = 0
        kernel[:, :, k, k] = kernel_2d

    return kernel
