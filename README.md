# FastMaskRCNN_ForwardTest

This is a __forward test__ script of a __single input image__ for the FastMaskRCNN: https://github.com/CharlesShang/FastMaskRCNN

![Display](https://github.com/MarkMoHR/FastMaskRCNN_ForwardTest/raw/master/assets/display2.png)

---
## Requirements
- [Tensorflow (>= 1.0.0)](https://www.tensorflow.org/install/install_linux)
- [Numpy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)
- [PIL (Pillow version = 2.3.0)](http://pythonware.com/products/pil/)
- [Pre-trained Resnet50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)
- [Pre-trained FastMaskRCNN model](https://drive.google.com/open?id=0B0J4gcV0gfL4U1NadkllSndKbFk) (From: [FastMaskRCNN #107](https://github.com/CharlesShang/FastMaskRCNN/issues/107#issuecomment-325446700))

## Functionalities
- Draw __bounding box__ of the predicted RoI
- Draw __mask__ of the predicted RoI (without bbox)
- Draw __mask__ of the predicted RoI with its bbox and predicted categroy

## Getting Start
1. It requires you to download the whole repo firstly from https://github.com/CharlesShang/FastMaskRCNN
2. Add the whole `./forward_test` under the root of the repo; then __replace__ the original `./libs/visualization/pil_utils.py` with the new one in my repo. Downloading the 2 pre-trained models above and place them(with the `./output/mask_rcnn/checkpoint`) as shown below. Finally the folder structure would be:
```
root/
├── data/pretrained_models/resnet_v1_50.ckpt
│   └── ......
│
├── forward_test/
│   ├── testdata/
│   ├── output/
│   └── forward_test_single_image.py
│
├── libs/
│   ├── visualization/
│   │   ├── pil_utils.py
│   │   └── ......
│   ├── nets/
│   │   ├── pyramid_network.py
│   │   └── ......
│   └── ......
│
├── output/mask_rcnn/
│   ├── checkpoint
│   ├── coco_resnet50_model.ckpt-2499999.data-00000-of-00001
│   ├── coco_resnet50_model.ckpt-2499999.index
│   └── coco_resnet50_model.ckpt-2499999.meta
└── ......

```
3. Modify original `./libs/nets/pyramid_network.py` according to [Issues#1-F3](https://github.com/MarkMoHR/FastMaskRCNN_ForwardTest/issues/1#issuecomment-354275222) and [Issues#1-F4](https://github.com/MarkMoHR/FastMaskRCNN_ForwardTest/issues/1#issuecomment-354277301). Or you can just __replace__ the original one with mine.
4. Put your test image under `./forward_test/testdata/`
5. (Optional) If you want to change the output image dir, modify the code `./forward_test/forward_test_single_image.py` at Line30-31
```
save_dir_bbox = 'output/bbox/'
save_dir_mask = 'output/mask/'
```
6. (Optional) If your test image is in __PNG__ format, modify the code `./forward_test/forward_test_single_image.py` at Line32
```
file_pattern = 'jpg'    # or 'png'
```
7. run `./forward_test/forward_test_single_image.py` and wait for the result

## Acknowledgment
- The `./forward_test/forward_test_single_image.py` is modified from the original `./train/train.py` from [FastMaskRCNN](https://github.com/CharlesShang/FastMaskRCNN)
- The `./libs/visualization/pil_utils.py` is modified from [@chen1005](https://github.com/CharlesShang/FastMaskRCNN/issues/26#issuecomment-319184033)'s suggestion
- The pre-trained Mask-RCNN model from [@QtSignalProcessing](https://github.com/CharlesShang/FastMaskRCNN/issues/107#issuecomment-325446700)

## To Do
- Fix the bug of [Issues#1-F8](https://github.com/MarkMoHR/FastMaskRCNN_ForwardTest/issues/1#issuecomment-354413737)
- Add [DenseCRF](https://github.com/lucasb-eyer/pydensecrf)
