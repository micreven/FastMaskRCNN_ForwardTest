#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os, sys
import glob
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import gmtime, strftime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import libs.nets.nets_factory as network

import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1

from train.train_utils import get_var_list_to_restore

from PIL import Image
from libs.visualization.pil_utils import draw_bbox, draw_mask

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

testdata_base_dir = 'testdata/'
save_dir_bbox = 'output/bbox/'
save_dir_mask = 'output/mask/'
file_pattern = 'jpg'    # or 'png'

FLAGS = tf.app.flags.FLAGS
resnet50 = resnet_v1.resnet_v1_50

def restore(sess):
    """choose which param to restore"""
    if FLAGS.restore_previous_if_exists:
        try:
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            ###########
            restorer = tf.train.Saver()
            ###########

            ###########
            # not_restore = [ 'pyramid/fully_connected/weights:0',
            #                 'pyramid/fully_connected/biases:0',
            #                 'pyramid/fully_connected/weights:0',
            #                 'pyramid/fully_connected_1/biases:0',
            #                 'pyramid/fully_connected_1/weights:0',
            #                 'pyramid/fully_connected_2/weights:0',
            #                 'pyramid/fully_connected_2/biases:0',
            #                 'pyramid/fully_connected_3/weights:0',
            #                 'pyramid/fully_connected_3/biases:0',
            #                 'pyramid/Conv/weights:0',
            #                 'pyramid/Conv/biases:0',
            #                 'pyramid/Conv_1/weights:0',
            #                 'pyramid/Conv_1/biases:0',
            #                 'pyramid/Conv_2/weights:0',
            #                 'pyramid/Conv_2/biases:0',
            #                 'pyramid/Conv_3/weights:0',
            #                 'pyramid/Conv_3/biases:0',
            #                 'pyramid/Conv2d_transpose/weights:0',
            #                 'pyramid/Conv2d_transpose/biases:0',
            #                 'pyramid/Conv_4/weights:0',
            #                 'pyramid/Conv_4/biases:0',
            #                 'pyramid/fully_connected/weights/Momentum:0',
            #                 'pyramid/fully_connected/biases/Momentum:0',
            #                 'pyramid/fully_connected/weights/Momentum:0',
            #                 'pyramid/fully_connected_1/biases/Momentum:0',
            #                 'pyramid/fully_connected_1/weights/Momentum:0',
            #                 'pyramid/fully_connected_2/weights/Momentum:0',
            #                 'pyramid/fully_connected_2/biases/Momentum:0',
            #                 'pyramid/fully_connected_3/weights/Momentum:0',
            #                 'pyramid/fully_connected_3/biases/Momentum:0',
            #                 'pyramid/Conv/weights/Momentum:0',
            #                 'pyramid/Conv/biases/Momentum:0',
            #                 'pyramid/Conv_1/weights/Momentum:0',
            #                 'pyramid/Conv_1/biases/Momentum:0',
            #                 'pyramid/Conv_2/weights/Momentum:0',
            #                 'pyramid/Conv_2/biases/Momentum:0',
            #                 'pyramid/Conv_3/weights/Momentum:0',
            #                 'pyramid/Conv_3/biases/Momentum:0',
            #                 'pyramid/Conv2d_transpose/weights/Momentum:0',
            #                 'pyramid/Conv2d_transpose/biases/Momentum:0',
            #                 'pyramid/Conv_4/weights/Momentum:0',
            #                 'pyramid/Conv_4/biases/Momentum:0',]
            # vars_to_restore = [v for v in  tf.all_variables()if v.name not in not_restore]
            # restorer = tf.train.Saver(vars_to_restore)
            # for var in vars_to_restore:
            #     print ('restoring ', var.name)
            ############

            restorer.restore(sess, checkpoint_path)
            print('restored previous model %s from %s' \
                  % (checkpoint_path, FLAGS.train_dir))
            time.sleep(2)
            return
        except:
            print('--restore_previous_if_exists is set, but failed to restore in %s %s' \
                  % (FLAGS.train_dir, checkpoint_path))
            time.sleep(2)

    if FLAGS.pretrained_model:
        if tf.gfile.IsDirectory(FLAGS.pretrained_model):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.pretrained_model)
        else:
            checkpoint_path = FLAGS.pretrained_model

        if FLAGS.checkpoint_exclude_scopes is None:
            FLAGS.checkpoint_exclude_scopes = 'pyramid'
        if FLAGS.checkpoint_include_scopes is None:
            FLAGS.checkpoint_include_scopes = 'resnet_v1_50'

        vars_to_restore = get_var_list_to_restore()
        # for var in vars_to_restore:
        #     print ('restoring ', var.name)

        try:
            restorer = tf.train.Saver(vars_to_restore)
            restorer.restore(sess, checkpoint_path)
            print('Restored %d(%d) vars from %s' % (
                len(vars_to_restore), len(tf.global_variables()),
                checkpoint_path))
        except:
            print('Checking your params %s' % (checkpoint_path))
            raise


def forward_test_single_image():
    if not os.path.exists(save_dir_bbox):
        os.makedirs(save_dir_bbox)
    if not os.path.exists(save_dir_mask):
        os.makedirs(save_dir_mask)

    file_pathname = testdata_base_dir + '*.' + file_pattern
    image_paths = glob.glob(file_pathname)    #with .jpg/.png
    image_names = glob.glob(file_pathname)    #no .jpg/.png
    for i in range(len(image_paths)):
        image_names[i] = image_paths[i][len(testdata_base_dir):-4]

    print(image_paths)
    print(image_names)

    TEST_image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    im_shape = tf.shape(TEST_image)

    ## network
    logits, end_points, pyramid_map = network.get_network(FLAGS.network, TEST_image,
                                                          weight_decay=FLAGS.weight_decay, is_training=True)
    outputs = pyramid_network.build(end_points, im_shape[1], im_shape[2], pyramid_map,
                                    num_classes=81,
                                    base_anchors=9,
                                    is_training=True,
                                    gt_boxes=None, gt_masks=None,
                                    loss_weights=[0.2, 0.2, 1.0, 0.2, 1.0])

    input_image = end_points['input']
    print("input_image.shape", input_image.shape)
    final_box = outputs['final_boxes']['box']
    final_cls = outputs['final_boxes']['cls']
    final_prob = outputs['final_boxes']['prob']
    final_mask = outputs['mask']['mask']

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    sess.run(init_op)

    ## restore trained model
    restore(sess)

    ## main loop
    coord = tf.train.Coordinator()
    threads = []
    # print (tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS))
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

    tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(len(image_paths)):
        ## read test image
        test_img = Image.open(image_paths[i])
        test_img = test_img.convert("RGB")
        test_img_i = np.array(test_img, dtype=np.uint8)
        test_img = np.array(test_img, dtype=np.float32)
        test_img = test_img[np.newaxis, ...]
        print("test_img.shape", test_img.shape)
        print("test_img_i.shape", test_img_i.shape)


        # start_time = time.time()

        input_imagenp, final_boxnp, final_clsnp, final_probnp, \
        final_masknp= \
            sess.run([input_image] + [final_box] + [final_cls] + [final_prob] +
                     [final_mask], feed_dict={TEST_image:test_img})

        # duration_time = time.time() - start_time

        draw_bbox(test_img_i,
                  type='est',
                  bbox=final_boxnp,
                  label=final_clsnp,
                  prob=final_probnp,
                  gt_label=None,
                  save_dir=save_dir_bbox,
                  save_name=image_names[i],
                  )

        print("final_masknp.shape\n", final_masknp.shape)

        draw_mask(test_img_i,
                  type='est',
                  bbox=final_boxnp,
                  mask=final_masknp,
                  label=final_clsnp,
                  prob=final_probnp,
                  gt_label=None,
                  save_dir=save_dir_mask,
                  save_name=image_names[i],
                  )

        if coord.should_stop():
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    forward_test_single_image()
