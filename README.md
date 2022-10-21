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
```

## Usage
modify tensorRT engine network size in [retina-engine-config](https://github.com/mrscarletzhou/retinaface-deepstream-python/blob/main/retina_network_config.txt) file
```
python3 main.py {VideoPath}
```

## References
+ [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)
+ [NVIDIA-AI-IOT/deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)


