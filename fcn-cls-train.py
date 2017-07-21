#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import logging
from tqdm import tqdm
from skimage import measure
from random import randint
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import picpac
import fcn_cls_nets
from gallery import Gallery

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db_fcn', 'db', 'training dataset')
flags.DEFINE_string('mixin_fcn', 'dbmixin', 'mixin negative dataset')

flags.DEFINE_string('db_cls', 'db_cls', 'training dataset')
flags.DEFINE_string('mixin_cls', 'wei_bg', 'mixin negative dataset')

flags.DEFINE_string('model', 'model_fcn_cls', 'Directory to put the training data.')
flags.DEFINE_string('net', 'resnet_tiny', '')
flags.DEFINE_string('val', 'db_val', '')

flags.DEFINE_string('opt', 'adam', '')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_bool('decay', True, '')
flags.DEFINE_float('decay_rate', 0.9, '')
flags.DEFINE_float('decay_steps', 10000, '')
flags.DEFINE_float('momentum', 0.99, 'when opt==mom')
flags.DEFINE_string('resume', None, '')
flags.DEFINE_integer('max_steps', 200000, '')
flags.DEFINE_integer('epoch_steps', 100, '')
flags.DEFINE_integer('val_epochs', 200, '')
flags.DEFINE_integer('ckpt_epochs', 20, '')
#flags.DEFINE_string('log', None, 'tensorboard')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_string('padding', 'SAME', '')
flags.DEFINE_integer('verbose', logging.INFO, '')
flags.DEFINE_float('pos_weight', None, '')
flags.DEFINE_integer('max_size', None, '')
flags.DEFINE_string('val_plot', None, '')
flags.DEFINE_integer('max_to_keep', 1000, '')
flags.DEFINE_float('contour_th', 0.5, '')
MAX_SAMPLES = 100


def logits2prob (v, scope='logits2prob', scale=None):
    with tf.name_scope(scope):
        shape = tf.shape(v)    # (?, ?, ?, 2)
        # softmax
        v = tf.reshape(v, (-1, 2))
        v = tf.nn.softmax(v)
        v = tf.reshape(v, shape)
        # keep prob of 1 only
        v = tf.slice(v, [0, 0, 0, 1], [-1, -1, -1, -1])
        # remove trailing dimension of 1
        v = tf.squeeze(v, axis=[3])
        if scale:
            v *= scale
    return v

def fcn_loss (logits, labels):
    # to HWC
    logits = tf.reshape(logits, (-1, 2))
    labels = tf.reshape(labels, (-1,))
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.to_int32(labels))
    if FLAGS.pos_weight:
        POS_W = tf.pow(tf.constant(FLAGS.pos_weight, dtype=tf.float32),
                       labels)
        xe = tf.multiply(xe, POS_W)
    loss = tf.reduce_mean(xe, name='fcn_xe')
    return loss, [loss] #, [loss, xe, norm, nz_all, nz_dim]

def cls_loss (logits, labels):
    # to HWC
    logits = tf.reshape(logits, (-1, 2))
    labels = tf.to_int32(tf.reshape(labels, (-1,)))
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    loss = tf.reduce_mean(xe, name='cls_xe')
    acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1, name="accuracy"), tf.float32), name='cls_acc')
    return loss, [loss, acc] #, [loss, xe, norm, nz_all, nz_dim]

def save_vis (path, prob, prob_cls, images):
    if images.shape[3] == 1:
        image = images[0, :, :, 0]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = images[0]
        pass

    prob = prob[0]
    contours = measure.find_contours(prob, FLAGS.contour_th)

    prob *= 255
    prob = cv2.cvtColor(prob, cv2.COLOR_GRAY2BGR)
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)

    H = max(image.shape[0], prob.shape[0])
    both = np.zeros((H, image.shape[1]*2 + prob.shape[1], 3))
    both[0:image.shape[0],0:image.shape[1],:] = image
    off = image.shape[1]

    for contour in contours:
        tmp = np.copy(contour[:,0])
        contour[:, 0] = contour[:, 1]
        contour[:, 1] = tmp
        contour = contour.reshape((1, -1, 2)).astype(np.int32)
        cv2.polylines(image, contour, True, (0, 255, 0))
        cv2.polylines(prob, contour, True, (0, 255, 0))
    cv2.putText(prob, '.3f' % prob_cls, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0))

    both[0:image.shape[0],off:(off+image.shape[1]),:] = image
    off += image.shape[1]
    both[0:prob.shape[0],off:(off+prob.shape[1]),:] = prob
    cv2.imwrite(path, both)

def main (_):
    logging.basicConfig(level=FLAGS.verbose)
    try:
        os.makedirs(FLAGS.model)
    except:
        pass
    assert FLAGS.db_fcn and os.path.exists(FLAGS.db_fcn)
    assert FLAGS.db_cls and os.path.exists(FLAGS.db_cls)

    X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
    Y_fcn = tf.placeholder(tf.float32, shape=(None, None, None, 1), name="labels_fcn")
    Y_cls = tf.placeholder(tf.float32, shape=(None,), name="labels_cls")

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d],
                            padding=FLAGS.padding):
        logits_fcn, logits_cls, stride = getattr(fcn_cls_nets, FLAGS.net)(X)
    loss_fcn, metric_fcn = fcn_loss(logits_fcn, Y_fcn)
    _ = tf.identity(logits_fcn, name='logits')	# make compatible with fcn-val.py
    loss_cls, metric_cls = cls_loss(logits_cls, Y_cls)
    prob_fcn = logits2prob(logits_fcn)
    prob_cls = tf.nn.softmax(logits_cls)
    #tf.summary.scalar("loss", loss)
    metric_names_fcn = [x.name[:-2] for x in metric_fcn]

    metric_names_cls = [x.name[:-2] for x in metric_cls]

    rate = FLAGS.learning_rate
    if FLAGS.opt == 'adam':
        rate /= 100
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if FLAGS.decay:
        rate = tf.train.exponential_decay(rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    if FLAGS.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(rate)
    elif FLAGS.opt == 'mom':
        optimizer = tf.train.MomentumOptimizer(rate, FLAGS.momentum)
    else:
        optimizer = tf.train.GradientDescentOptimizer(rate)
        pass

    train_op_fcn = optimizer.minimize(loss_fcn, global_step=global_step)
    train_op_cls = optimizer.minimize(loss_cls, global_step=global_step)

    picpac_config_shared = dict(seed=2016,
                loop=True,
                perturb=True,
                shuffle=True,
                reshuffle=True,
                max_size = 300,
                #resize_width=256,
                #resize_height=256,
                batch=1,
                pert_angle=0,
                pert_hflip=True,
                pert_vflip=False,
                pert_color1=10,
                pert_color2=10,
                pert_color3=10,
                pert_min_scale = 0.8,
                pert_max_scale = 1.2,
                channels=FLAGS.channels,
                #mixin = FLAGS.mixin,
                stratify=True,
                #pad=False,
                channel_first=False, # this is tensorflow specific
                )
    picpac_config_fcn = dict(
                round_div = stride,
                annotate='json',
                mixin_group_delta=1,
		)
    picpac_config_fcn.update(picpac_config_shared)
    if FLAGS.mixin_fcn:
        picpac_config_fcn['mixin'] = FLAGS.mixin_fcn

    picpac_config_cls = picpac_config_shared
    # picpac_config_cls.update(picpac_config_shared)
    if FLAGS.mixin_cls:
        picpac_config_cls['mixin'] = FLAGS.mixin_cls
    # print(picpac_config_cls)

    stream_fcn = picpac.ImageStream(FLAGS.db_fcn, **picpac_config_fcn)
    stream_cls = picpac.ImageStream(FLAGS.db_cls, **picpac_config_cls)
    val_stream = None
    if FLAGS.val and FLAGS.val_plot:
        assert os.path.exists(FLAGS.val)
        val_stream = picpac.ImageStream(FLAGS.val, perturb=False, loop=False, **fg_config)


    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3


    with tf.Session(config=config) as sess:
        sess.run(init)
        if FLAGS.resume:
            saver.restore(sess, FLAGS.resume)
        step = 0
        epoch = 0
        global_start_time = time.time()
        while step < FLAGS.max_steps:
            start_time = time.time()
            avg_fcn = np.array([0] * len(metric_fcn), dtype=np.float32)
            avg_cls = np.array([0] * len(metric_cls), dtype=np.float32)
            for _ in tqdm(range(FLAGS.epoch_steps), leave=False):
		# train FCN
                images, labels, _ = stream_fcn.next()
                #print('xxx', images.shape)
                feed_dict = {X: images, Y_fcn: labels}
                mm, _ = sess.run([metric_fcn, train_op_fcn], feed_dict=feed_dict)
                avg_fcn += np.array(mm)
		# train CLS
                images, labels, _ = stream_cls.next()
                #print('yyy', images.shape, labels)
                feed_dict = {X: images, Y_cls: labels}
                mm, _ = sess.run([metric_cls, train_op_cls], feed_dict=feed_dict)
                avg_cls += np.array(mm)
                step += 1
                pass
            avg_fcn /= FLAGS.epoch_steps
            avg_cls /= FLAGS.epoch_steps
            stop_time = time.time()
            txt_fcn = ', '.join(['%s=%.4f' % (a, b) for a, b in zip(metric_names_fcn, list(avg_fcn))])
            txt_cls = ', '.join(['%s=%.4f' % (a, b) for a, b in zip(metric_names_cls, list(avg_cls))])
            print('step %d: elapsed=%.4f time=%.4f, %s %s'
                    % (step, (stop_time - global_start_time), (stop_time - start_time), txt_fcn, txt_cls))
            epoch += 1
            if epoch and (epoch % FLAGS.ckpt_epochs == 0):
                ckpt_path = '%s/%d' % (FLAGS.model, step)
                start_time = time.time()
                saver.save(sess, ckpt_path)
                stop_time = time.time()
                print('epoch %d step %d, saving to %s in %.4fs.' % (epoch, step, ckpt_path, stop_time - start_time))
            if epoch and (epoch % FLAGS.val_epochs == 0) and val_stream:
                val_stream.reset()
                #avg = np.array([0] * len(metrics), dtype=np.float32)
                gal = Gallery(os.path.join(FLAGS.val_plot, str(step)))
                for images, _, _ in val_stream:
                    feed_dict = {X: images}
                    #print("XXX", images.shape)
                    pp_fcn, pp_cls = sess.run([prob_fcn, prob_cls, metrics], feed_dict=feed_dict)
                    save_vis(gal.next(), pp_fcn, pp_cls, images)
                gal.flush()
                print('epoch %d step %d, validation')
            pass
        pass
    pass

if __name__ == '__main__':
    tf.app.run()

