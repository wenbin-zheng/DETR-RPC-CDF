# Weed-DETR
This repo is the implementation of 《Lightweight Weed Detection
 Model Based on Re-parameterized Partial Convolution and Multi-Scale Feature Fusion》.This study proposes a lightweight model for weed detection named  Weed DEtection TRansformer(Weed-DETR).  We propose a novel convolution for backbone networks called re-parameterized partial convolution (RP-Conv).In the feature fusion stage, we introduce the Collection-Distribution Feature Fusion (CDFF) mechanism. Our proposed method meets the requirements for accuracy and lightweight design in weed identification, providing the necessary technical support for real-time weed detection in the field.

# Install
```bash
$ git clone https://github.com/wenbin-zheng/Weed-DETR.git
$ cd 
$ pip install -r requirements.txt
```
<summary>Install</summary>

[**Python>=3.8.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including
[**PyTorch>=1.8**](https://pytorch.org/get-started/locally/):



# Updates

* Upload weed-detr.pt and a portion of the code.

# Inference

* `Datasets` : [CropAndWeed](https://github.com/cropandweed/cropandweed-dataset)


```bash
$ python val.py --data dataset/data.yaml --imgsz=640
```
# Train
train.py allows you to train new model from strach.
```bash
$ python train.py --batch 4 --imgsz 640 --data dataset/data.yaml  --device 0 --epochs 250
```
```
```

# References
Thanks to their great works
* [RT-DETR](https://github.com/lyuwenyu/RT-DETR)

  
