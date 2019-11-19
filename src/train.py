## train
import tensorflow as tf
from config import *
import os
from imdb import kitti


FLAGS =tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset','KITTI',"""Currently only support KITTI dataset """)
tf.app.flags.DEFINE_string( 'gpu',default='0', """gpu id"""  )
tf.app.flags.DEFINE_string('net','RSnet',""" Neural net architecture""")
tf.app.flags.DEFINE_string('pretrained_model_path', '',"""path to the pretrained model""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'train', """"can be train, trainval, val, or test""")

def train():
    assert FLAGS.dataset == 'KITTI',\
    'Currently only support KITTI dataset'

    os.environ['CUDA_VISIBLE_DEVICES'] =FLAGS.gpu

    with tf.Graph().as_default():
        assert FLAGS.net == 'RSnet',\
        'Selected neutral net architecture not supported : {}'.format(FLAGS.net)

        if FLAGS.net == 'RSnet':
            mc = kitti_RSnet_config()
            mc.PRETRAINED_MODEL_PATH =FLAGS.pretrained_model_path
            model = RSnet(mc)

        imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)
