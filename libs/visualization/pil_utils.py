import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from scipy.misc import imresize

FLAGS = tf.app.flags.FLAGS
_DEBUG = False

PROB_THRESHOLD = 0.7
PIXEL_FILLED_THRESHOLD = 0.85


def draw_img(step, image, name='', image_height=1, image_width=1, rois=None):
    # print("image")
    # print(image)
    # norm_image = np.uint8(image/np.max(np.abs(image))*255.0)
    norm_image = np.uint8(image / 0.1 * 127.0 + 127.0)
    # print("norm_image")
    # print(norm_image)
    source_img = Image.fromarray(norm_image)
    return source_img.save(FLAGS.train_dir + 'test_' + name + '_' + str(step) + '.jpg', 'JPEG')

def draw_bbox(image, type='', image_height=1, image_width=1, bbox=None, label=None, gt_label=None,
                  prob=None, save_dir='', save_name=''):
    #print(prob[:,label])
    source_img = Image.fromarray(image)
    # b, g, r = source_img.split()
    # source_img = Image.merge("RGB", (r, g, b))
    draw = ImageDraw.Draw(source_img)
    color = '#0000ff'
    if bbox is not None:
        for i, box in enumerate(bbox):
            if label is not None:
                if prob is not None:
                    if (prob[i,label[i]] > PROB_THRESHOLD) and (label[i] > 0):
                        if gt_label is not None:
                            text  = cat_id_to_cls_name(label[i]) + ' : ' + cat_id_to_cls_name(gt_label[i])
                            if label[i] != gt_label[i]:
                                color = '#ff0000'#draw.text((2+bbox[i,0], 2+bbox[i,1]), cat_id_to_cls_name(label[i]) + ' : ' + cat_id_to_cls_name(gt_label[i]), fill='#ff0000')
                            else:
                                color = '#0000ff'
                        else:
                            text = cat_id_to_cls_name(label[i])

                        draw.text((2+bbox[i,0], 2+bbox[i,1]), text, fill=color)

                        if _DEBUG is True:
                            print("plot",label[i], prob[i,label[i]])

                        draw.rectangle(box,fill=None,outline=color)
                    else:
                        if label[i] > 0:
                            print("skip", prob[i,label[i]])
                else:
                    text = cat_id_to_cls_name(label[i])
                    draw.text((2+bbox[i,0], 2+bbox[i,1]), text, fill=color)
                    draw.rectangle(box,fill=None,outline=color)

    return source_img.save(save_dir + type + '_' + save_name + '.png', 'PNG')

def draw_mask(image, type='', bbox=None, mask=None, label=None, gt_label=None,
              prob=None, save_dir='', save_name=''):
    # print(prob[:,label])
    source_img = Image.fromarray(image)
    source_img_with_box = Image.fromarray(image)
    draw = ImageDraw.Draw(source_img_with_box)
    color = '#0000ff'
    if bbox is not None:
        for i, box in enumerate(bbox):
            if label is not None:
                mask_color_id = np.random.randint(15)
                box1 = np.floor(box).astype('uint16')
                box_w = box1[2] - box1[0]
                box_h = box1[3] - box1[1]

                if prob is not None:
                    if (prob[i, label[i]] > PROB_THRESHOLD) and (label[i] > 0):

                        if mask is not None:
                            m = np.array(mask * 255.0)
                            m = np.transpose(m, (0, 3, 1, 2))

                            color_img = color_id_to_color_code(mask_color_id) * np.ones((box_h, box_w, 1)) * 255
                            color_img = Image.fromarray(color_img.astype('uint8')).convert('RGB')

                            resized_m = imresize(m[i][label[i]], [box_h, box_w], interp='bilinear')  # label[i]
                            resized_m[resized_m >= 128] = 128
                            resized_m[resized_m < 128] = 0

                            # print("#### resized_m.shape", resized_m.shape)
                            # print(resized_m)
                            n_filled_pixs = np.sum(resized_m) / 128
                            pixel_filled_p = float(n_filled_pixs / box_h / box_w)
                            print("#### n_filled_pixs", n_filled_pixs, "total", box_h*box_w,
                                  pixel_filled_p ,"%")
                            if pixel_filled_p < PIXEL_FILLED_THRESHOLD:
                                resized_m = Image.fromarray(resized_m.astype('uint8'), 'L')
                                source_img.paste(color_img, (box1[0], box1[1]), mask=resized_m)
                                source_img_with_box.paste(color_img, (box1[0], box1[1]), mask=resized_m)

                                text = cat_id_to_cls_name(label[i])
                                draw.text((2 + bbox[i, 0], 2 + bbox[i, 1]), text, fill=color)
                                draw.rectangle(box, fill=None, outline=color)

    source_img.save(save_dir + type + '_no_box_' + save_name + '.png', 'PNG')
    source_img_with_box.save(save_dir + type + '_with_box_' + save_name + '.png', 'PNG')


def cat_id_to_cls_name(catId):
    cls_name = np.array(['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                         'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                         'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                         'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                         'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                         'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                         'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                         'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                         'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                         'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                         'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                         'scissors', 'teddy bear', 'hair drier', 'toothbrush'])
    return cls_name[catId]


def color_id_to_color_code(colorId):
    color_code = np.array([[178, 31, 53],
                           [216, 39, 53],
                           [255, 116, 53],
                           [255, 161, 53],
                           [255, 203, 53],
                           [255, 255, 53],
                           [0, 117, 58],
                           [0, 158, 71],
                           [22, 221, 53],
                           [0, 82, 165],
                           [0, 121, 231],
                           [0, 169, 252],
                           [104, 30, 126],
                           [125, 60, 181],
                           [189, 122, 246]])
    return color_code[colorId]