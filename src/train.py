## train
import tensorflow as tf
from config import *
import os
from imdb import kitti
import threading
import time
from six.moves import xrange

from utils.util import *
from nets import *




FLAGS =tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset','KITTI',"""Currently only support KITTI dataset """)
tf.app.flags.DEFINE_string( 'gpu',default='0', """gpu id"""  )
tf.app.flags.DEFINE_string('net','RSnet',""" Neural net architecture""")
tf.app.flags.DEFINE_string('pretrained_model_path', '',"""path to the pretrained model""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'train', """"can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('train_dir','', """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_string('summary_step', 50, """Number of steps to save summary""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Maximum number of batches to run.""")

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

        imbd = kitti(FLAGS.image_set, FLAGS.data_path, mc)

        # save model size, flops, activations by layers
        with open(os.path.join(FLAGS.train_dir, 'train_dir', 'model_metrics.txt'), 'w') as f:
            f.write ('Number of parameter by layer:\n')
            count = 0
            for c in model.model_size_counter:
                f.write('\t{}:{}\n'.format(c[0], c[1]))
                count += c[1]
            f.write('\ttotal: {}\n'.format(count))

            count = 0
            f.write('\nActivation size by Layer: \n')
            for c in model.activation_counter:
                f.write('\ttotal: {}\n'.format(c[0], c[1]))
                count += c[1]
            f.write('\ttotal: {}\n'.format(count))

            count  = 0
            f.write('\nNumber of flops by layer: \n')
            for c in model.flop_counter:
                f.write('\n{}:{}\n'.format(c[0], c[1]))
                count += c[1]
                f.write('\ttotal: {}\n'.format(count))
        f.close()
        print('Model statistics saved to {}.'.format(
            os.path.join(FLAGS.train_dir, 'model_metrics.txt')))

        def enqueue(sess, coord):
            with coord.stop_on_exception():
                while not coord.should_stop():
                    #read batch input
                    lidar_per_batch, lidar_mask_per_batch, label_per_batch, \
                        weight_per_batch = imbd.read_batch()
                    feed_dict = {
                        model.ph_keep_prob : mc.KEEP_PROB,
                        model.ph_lidar_input: lidar_per_batch,
                        model.ph_lidar_mask: lidar_mask_per_batch,
                        model.ph_lable: label_per_batch,
                        model.ph_loss_weight: weight_per_batch
                    }

                    sess.run(model.enqueue_op, feed_dict)

        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.summary.merge_all() ## save all summary and can be showed in tensorboard
        init = tf.initialize_all_variables()

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        ##threads
        coord = tf.train_Coordinator()
        enq_threads = []
        for _ in range(mc.NUM_ENQUEUE_THREAD):
            eqth = threading.Thread(target=enqueue, args=[sess, coord])
            eqth.start()
            enq_threads.append(eqth)

        run_options = tf.RunOptions(timeout_in_ms=60000) #todo

        try:
            for step in xrange(FLAGS.max_steps):
                start_time = time.time()

                if step % FLAGS.summary_step == 0 or step == FLAGS.max_steps-1 :
                    op_list =[
                        model.lidar_input, model.lidar_mask, model.label, model.train_op, model.loss,
                        model.pred_cls, summary_op
                    ]

                    lidar_per_batch, lidar_mask_per_batch, label_per_batch, \
                    _, loss_value, pred_cls, summary_str = sess.run(op_list, options=run_options)








