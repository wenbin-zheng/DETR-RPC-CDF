# DETR-RPC-CDF
This repo is the implementation of "Lightweight Weed Detection using Re-parameterized Partial Convolution and Collection-Distribution Feature Fusion". This paper proposes a lightweight DEtection TRansformer model (DETR-RPC-CDF) with Re-parameterized Partial Convolution (RPC) and Collection-Distribution feature Fusion (CDF) mechanism.  This model employs Re-parameterized Partial Convolution (RPC) in the backbone network to reduce computational redundancy and enhance detection speed.   Additionally, This model introduces a Collection-Distribution Feature Fusion (CDF) mechanism to reduce information loss during feature integration and preserve multi-scale details.

# Install
```bash
$ git clone https://github.com/wenbin-zheng/DETR-RPC-CDF.git
$ cd Weed-DETR
$ pip install -r requirements.txt
```
<summary>Install</summary>

[**Python>=3.8.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including
[**PyTorch>=1.8**](https://pytorch.org/get-started/locally/):




# Inference

* `Datasets` : [CropAndWeed](https://github.com/cropandweed/cropandweed-dataset)


```bash
$ python val.py --data dataset/data.yaml --imgsz=640
```
# Train
train.py allows you to train new model.
```bash
$ python train.py --batch 4 --imgsz 640 --data dataset/data.yaml  --device 0 --epochs 250
```

# References
Thanks to their great works
* [RT-DETR](https://github.com/lyuwenyu/RT-DETR)

  
