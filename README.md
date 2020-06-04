# Coral BodyPix

BodyPix is an open-source machine learning model which allows for person and
body-part segmentation. This has previously been released as a
[Tensorflow.Js](https://blog.tensorflow.org/2019/11/updated-bodypix-2.html)
project.

This repo contains a set of BodyPix Models (both with MobileNet v1 and ResNet
50 backbones) that are quantized and optimized for the Coral Edge TPU.
Example code is provided to enable generic platforms as well as an optimized
version for the Coral Dev Board.

## What is Person/Body-Part Segmentation?

Image segmentation refers to grouping pixels of an image into semantic areas,
typically to locate objects and boundaries. For example, the Coral DeepLab
model (available on the [Coral Models Page](https://coral.ai/models/)) segments
based on 20 objects. In this example, as with all segmentation examples, pixels
are classified as one of those objects or background.

BodyPix extends this concept and segments for people as well as twenty-four
body parts (such as "right hand" or "torso front"). More information can be
found on the
 [Tensorflow.Js](https://blog.tensorflow.org/2019/11/updated-bodypix-2.html)
page. This model and post-processing (contained as a custom OP in the Edge
TPU TFLite Interpreter) has been optimized for the Edge TPU.

## Examples in this repo

NOTE: BodyPix relies on the latest version of the Coral API and for the Dev
Board the latest Mendel system image.

To install all the requirements, simply run

```
sh install_requirements.sh
```

### bodypix.py

A generic BodyPix example intended to be run on multiple platforms, which has
not been optimized. Note that this is not recommended for the Coral Dev Board,
 where the performance is poor compared to the bodypix_gl_imx example. This
 example allows segmentation of a person, segmentation of body parts, as well
 as an anonymizer option which lets you remove the person from the camera
 image.

Run the base demo (using the MobileNet v1 backbone with 640x480 input) like
this:

```bash
python3 bodypix.py
```

To segment body parts (grouped as regions as opposed to displaying all 24)
 instead of the entire person, pass the `--bodyparts` flag:

```bash
python3 bodypix.py --bodyparts
```

In this repo we have included 11 BodyPix model files using different backbone
networks and supporting different input resolutions. There are significant
trade-offs in these versions, MobileNet will be faster than ResNet but
less accurate; larger resolutions are slower but allow a wider field of
view (allowing further-away people to be processed correctly).

This can be changed with the `--model` flag. The following models are
provided:

```bash
models/bodypix_mobilenet_v1_075_1024_768_16_quant_edgetpu_decoder.tflite
models/bodypix_mobilenet_v1_075_1280_720_16_quant_edgetpu_decoder.tflite
models/bodypix_mobilenet_v1_075_480_352_16_quant_edgetpu_decoder.tflite
models/bodypix_mobilenet_v1_075_640_480_16_quant_edgetpu_decoder.tflite
models/bodypix_mobilenet_v1_075_768_576_16_quant_edgetpu_decoder.tflite
models/bodypix_resnet_50_416_288_16_quant_edgetpu_decoder.tflite
models/bodypix_resnet_50_640_480_16_quant_edgetpu_decoder.tflite
models/bodypix_resnet_50_768_496_32_quant_edgetpu_decoder.tflite
models/bodypix_resnet_50_864_624_32_quant_edgetpu_decoder.tflite
models/bodypix_resnet_50_928_672_16_quant_edgetpu_decoder.tflite
models/bodypix_resnet_50_960_736_32_quant_edgetpu_decoder.tflite
```

You can change the camera resolution by using the `--width` and `--height`
parameter. Note that in general the camera resolution should equal or exceed
 the input resolution of the network to get the full advantage of the higher
 resolution inference:

```bash
python3 bodypix.py --width 480 --height 360  # fast but low res
python3 bodypix.py --width 640 --height 480  # default
python3 bodypix.py --width 1280 --height 720 # slower but high res
```

If the camera and monitor are both facing you, consider adding the `--mirror` flag:

```bash
python3 bodypix.py --mirror
```

If your input camera supports encoded frames (h264 or JPEG) you can provide
the corresponding flags to increase performance. Note these modes are mutually
exclusive:

```bash
python3 bodypix.py --h264
python3 bodypix.py --jpeg
```

You can enable Anonymizer mode (which anonymizes the person, similar to in the
[Coral PoseNet Project](https://github.com/google-coral/project-posenet). As
opposed to the PoseNet example, instead of indicating the pose skeleton the
entire outline of the person is indicated.

```bash
python3 bodypix.py --anonymize
```

### bodypix_gl_imx.py

This example is optimized specifically for the iMX8MQ GPU and VPU found on the
Coral Dev Board. It is intended to allow real time processing and rendering on
the platform (able to achieve 30 FPS even at 1280x720 resolution). The flags
 for input (models, camera configuration) are the same but we enable
 toggling between display modes with key presses instead of a flag:

```bash
python3 bodypix_gl_imx.py
```

The following key presses can be used to toggle various modes:

```bash
Toggle PoseNet-style Skeletons: 's'
Toggle Bounding Boxes: 'b'
Toggle Anonymizer: 'a'
Toggle Aggregated Heatmap Generation: 'h'
Toggle Body Part Segmentation: 'p'
Reset: 'r'
```



