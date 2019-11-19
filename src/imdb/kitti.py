"""Image data base class for kitti"""

import os
import numpy as np
import subprocess
from imbd import  imbd

class kitti(imbd):
    def __init__(self, image_set, data_path, mc):
        imbd.__init__(self, 'kitti_'+image_set, mc)
        self._image_set = image_set
        self._data_root_path = data_path
        self._lidar_2d_path = os.path.join(self._data_root_path, 'lidar_2d')
        self._gtd_2d_path = os.path.join(self._data_root_path, 'gta')

        #a list of string indices of images in the directory
        self._image_idx = self._load_image_set_idx()

    def _load_image_set_idx(self):
        image_set_file = os.path.join(
            self._data_root_path, 'ImageSet', self._image_set+'.txt')
        assert os.path.exists(image_set_file), \
            'File does not exist: {}'.format(image_set_file)

        with open (image_set_file) as f:
            image_idx 