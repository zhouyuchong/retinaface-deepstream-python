# RetinaFace with Deepstream 6.0
This is a face detection app build on Deepstream.
There are several repos about retinaface & deepstream. But none is compatible with the latest version of deepstream. So there it is.

## Requirements
+ Deepstream 6.0
+ GStreamer 1.14.5
+ Cuda 11.4+
+ NVIDIA driver 470.63.01+
+ TensorRT 8+

Follow [deepstream](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#dgpu-setup-for-ubuntu) official doc to install dependencies.

Deepstream docker is more recommended.
## Pretrained
Please refer to this [repo](https://github.com/wang-xinyu/tensorrtx) for pretrained models and serialized TensorRT engine.

## Installation
```
git clone https://github.com/mrscarletzhou/retinaface-deepstream-python.git
cd retinaface-deepstream-python
make nvdsinfer_customparser
```

## Usage
modify tensorRT engine network size in [retina-engine-config](https://github.com/mrscarletzhou/retinaface-deepstream-python/blob/main/retina_network_config.txt) file
```
LD_PRELOAD=./libnvdsinfer_custom_impl_Retinaface.so python3 main.py {VideoPath}
```

## References
+ [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)
+ [NVIDIA-AI-IOT/deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)

## Known issue
+ get a large amount of wrong outputs

That's because of the [sychronazation bug](https://github.com/wang-xinyu/tensorrtx/commit/e72d9db48ba8453fd4465048a0175621f1b1c501#diff-e4f7cf998c56a033573edc39c7736317f73a28402d835ee44001bac64f386dfb) of tensorrtx codes. To solve it you should modify the [decode.cu](https://github.com/wang-xinyu/tensorrtx/blob/master/retinaface/decode.cu) file in tensorrtx repo and regenerate the engine.

## Undone
+ ~~to pass the landmarks to probe and display them. (tring to use NvDsUserMeta)~~
+ ~~adapt scale of input-network and output~~
+ Modularity
+ multiple src input
+ dynamic add and delete
+ pause and continue function
