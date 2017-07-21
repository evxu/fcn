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
    def __init__ (self, path, name='logits_cls:0', prob=True):
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
            outputs = tf.nn.softmax(outputs)
            outputs = tf.slice(outputs, [0, 0, 0, 1], [-1, -1, -1, -1])
            # remove trailing dimension of 1
            outputs = tf.squeeze(outputs, axis=[1,2,3])
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
flags.DEFINE_integer('max', 100, '')
flags.DEFINE_string('name', 'logits_cls:0', '')
flags.DEFINE_integer('max_size', None, '')
flags.DEFINE_string('out', 'val/out', '')


def save(path, images):
    image = images[0,:,:,:]
    cv2.imwrite(path, image)



def main (_):
    assert FLAGS.db and os.path.exists(FLAGS.db)

    stream = picpac.ImageStream(FLAGS.db, 
        max_size=300,
        channel_first=False, 
        perturb=False, 
        loop=False, 
        channels=FLAGS.channels)

    gal = Gallery(FLAGS.out, score=True)
    cc = 0
    with Model(FLAGS.model, name=FLAGS.name, prob=True) as model:
        for images, _, _ in stream:
            #print(images.shape)
            #images *= 600.0/1500
            #images -= 800
            #images *= 3000 /(2000-800)
            probs = model.apply(images)
            print(probs)
            save(gal.next(score=probs[0]), images)
            
            cc += 1
            if FLAGS.max and cc>= FLAGS.max:
                break
    gal.flush(rank=True)
    pass

if __name__ == '__main__':
    tf.app.run()
