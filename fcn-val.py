#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from tqdm import tqdm
import numpy as np
import cv2
from skimage import measure
# RESNET: import these for slim version of resnet
import tensorflow as tf
import picpac
# from stitcher import Stitcher
from gallery import Gallery

class Model:
    def __init__ (self, path, name='logits:0', prob=False):
        """applying tensorflow image model.

        path -- path to model
        name -- output tensor name
        prob -- convert output (softmax) to probability
        """
        graph = tf.Graph()
        with graph.as_default():
            saver = tf.train.import_meta_graph(path + '.meta')
        if False:
            for op in graph.get_operations():
                for v in op.values():
                    print(v.name)
        inputs = graph.get_tensor_by_name("images:0")
        outputs = graph.get_tensor_by_name(name)
        if prob:
            shape = tf.shape(outputs)    # (?, ?, ?, 2)
            # softmax
            outputs = tf.reshape(outputs, (-1, 2))
            outputs = tf.nn.softmax(outputs)
            outputs = tf.reshape(outputs, shape)
            # keep prob of 1 only
            outputs = tf.slice(outputs, [0, 0, 0, 1], [-1, -1, -1, -1])
            # remove trailing dimension of 1
            outputs = tf.squeeze(outputs, axis=[3])
            pass
        self.prob = prob
        self.path = path
        self.graph = graph
        self.inputs = inputs
        self.outputs = outputs
        self.saver = saver
        self.sess = None
        pass

    def __enter__ (self):
        assert self.sess is None
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config, graph=self.graph)
        #self.sess.run(init)
        self.saver.restore(self.sess, self.path)
        return self

    def __exit__ (self, eType, eValue, eTrace):
        self.sess.close()
        self.sess = None

    def apply (self, images, batch=32):
        if self.sess is None:
            raise Exception('Model.apply must be run within context manager')
        if len(images.shape) == 3:  # grayscale
            images = images.reshape(images.shape + (1,))
            pass

        return self.sess.run(self.outputs, feed_dict={self.inputs: images})
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'db', '')
flags.DEFINE_string('model', 'model', 'Directory to put the training data.')
flags.DEFINE_integer('channels', 3, '')  # changed from 1 to 3  --Evelyn
flags.DEFINE_integer('patch', None, '')
flags.DEFINE_string('out', None, '')
flags.DEFINE_integer('max', 100, '')
flags.DEFINE_string('name', 'logits:0', '')
flags.DEFINE_float('cth', 0.5, '')
flags.DEFINE_float('fraction', 256, 'fraction of the number of pixels')
flags.DEFINE_integer('stride', 1, '')
flags.DEFINE_integer('max_size', None, '')


def save (path, images, prob):
    ### image = images[0, :, :, 0]
    # image = cv2.cvtColor(images[0, :, :, :], cv2.COLOR_RGB2GRAY)
    image = images[0, :, :, :]
    prob = prob[0]
    contours = measure.find_contours(prob, FLAGS.cth)

    
    ### add temp to fix incompatibility
    # temp = np.zeros((image.shape[0], image.shape[1]))
    # cv2.normalize(image, temp, 0, 255, cv2.NORM_MINMAX)
    # image = temp

    H = max(image.shape[0], prob.shape[0])
    both = np.zeros((H, image.shape[1]*2 + prob.shape[1], 3))
    both[0:image.shape[0],0:image.shape[1],:] = image
    off = image.shape[1]

    # draw bounding boxes
    binary = np.array(prob > FLAGS.cth, dtype=np.uint8)
    labels = measure.label(binary)
    properties = measure.regionprops(labels)


    total_pixel = image.shape[0]*image.shape[1]
    th = total_pixel/FLAGS.fraction
    for region in properties:
        if region.area > th:
            bb = region.bbox
            cv2.rectangle(image, (bb[1], bb[0]), (bb[3], bb[2]), [0,0,255], 2)
    
    prob *= 255
    for contour in contours:
        tmp = np.copy(contour[:,0])
        contour[:, 0] = contour[:, 1]
        contour[:, 1] = tmp
        contour = contour.reshape((1, -1, 2)).astype(np.int32)
        # cv2.polylines(image, contour, True, [0,0,255], 2)
        cv2.polylines(prob, contour, True, 255)

    both[0:image.shape[0],off:(off+image.shape[1]),:] = image
    off += image.shape[1]
    both[0:prob.shape[0],off:(off+prob.shape[1]),:] = cv2.cvtColor(prob, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(path, both)


def main (_):
    assert FLAGS.out
    assert FLAGS.db and os.path.exists(FLAGS.db)

    picpac_config = dict(seed=2016,
                #loop=True,
                shuffle=True,
                reshuffle=True,
                max_size = 400,
                #resize_width=256,
                #resize_height=256,
                round_div = FLAGS.stride,
                batch=1,
                split=1,
                split_fold=0,
                annotate='json',
                channels=FLAGS.channels,
                stratify=True,
                pert_color1=20,
                pert_angle=20,
                pert_min_scale=0.4,
                pert_max_scale=0.6,
                #pad=False,
                pert_hflip=True,
                pert_vflip=True,
                channel_first=False # this is tensorflow specific
                                    # Caffe's dimension order is different.
                )

    stream = picpac.ImageStream(FLAGS.db, perturb=False, loop=False, **picpac_config)


    gal = Gallery(FLAGS.out, score=True)
    cc = 0
    with Model(FLAGS.model, name=FLAGS.name, prob=True) as model:
        for images, _, _ in stream:
            #images *= 600.0/1500
            #images -= 800
            #images *= 3000 /(2000-800)
            _, H, W, _ = images.shape
            if FLAGS.max_size:
                if max(H, W) > FLAGS.max_size:
                    continue
            print(images.shape)
            if FLAGS.patch:
                
                stch = Stitcher(images, FLAGS.patch)
                probs = stch.stitch(model.apply(stch.split()))
            else:
                probs = model.apply(images)
            cc += 1
            save(gal.next(score=0), images, probs)
            if FLAGS.max and cc >= FLAGS.max:
                break
    gal.flush(rank=True)
    pass

if __name__ == '__main__':
    tf.app.run()

# python fcn-val.py --db bag_style.test --model ./model/160000 --out val_bag_style --max 500

# python fcn-val.py --db true-bags --model ./model/160000 --out val_bag_ids --max 500
