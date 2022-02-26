

## 1. 简介
**本项目以PaddleDetection框架和YY-YOLOv2模型实现"人脸检测"**
>[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/)为基于飞桨PaddlePaddle的端到端目标检测套件，提供多种主流目标检测、实例分割、跟踪、关键点检测算法，配置化的网络模块组件、数据增强策略、损失函数等，推出多种服务器端和移动端工业级SOTA模型，并集成了模型压缩和跨平台高性能部署能力,帮助开发者更快更好完成端到端全开发流程。
>
>[PP-YOLO](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/configs/ppyolo/README_cn.md)是PaddleDetection优化和改进的YOLOv3的模型，其精度(COCO数据集mAP)和推理速度均优于YOLOv4模型。
>![](https://ai-studio-static-online.cdn.bcebos.com/cfeb4d56c11c4b31a92b5d2e15231e4aca01ddb84ecc4f2c86c698f9f06d6ddb)
 
## 2. 数据集介绍
>[WIDER FACE](http://shuoyang1213.me/WIDERFACE/)数据集是人脸检测的一个benchmark数据集，包含32203图像，以及393,703个标注人脸，其中，158,989个标注人脸位于训练集，39,,496个位于验证集。每一个子集都包含3个级别的检测难度：Easy，Medium，Hard。这些人脸在尺度，姿态，光照、表情、遮挡方面都有很大的变化范围。
>![](https://ai-studio-static-online.cdn.bcebos.com/c89b7104f86c4bd4ae1152e5472c0f8a2e37a311bbd54d9bb1cf40f1a1f4c5a7)
>
>WIDER FACE选择的图像主要来源于公开数据集WIDER。制作者来自于香港中文大学，他们选择了WIDER的61个事件类别，对于每个类别，随机选择40%10%50%作为训练、验证、测试集。
也可以也在AI Studio上下载[数据集](https://aistudio.baidu.com/aistudio/datasetdetail/4336),

## 3. 开始使用

### 3.1 准备环境

- 硬件： Tesla V100 * 1
- 框架：
  - PaddlePaddle
  - PaddleDetection

将本项目git clone之后进入项目，使用`pip install -r requirements.txt`安装依赖,
再用`python setup.py install`即可。

### 3.2 快速开始

#### 第一步：克隆本项目

```bash
# clone this repo
git clone https://github.com/tianxingxia-cn/PaddleFaceDectection.git
cd PaddleFaceDectection
```
#### 第二步：安装依赖项
```bash
pip install -r requirements.txt
python setup.py install
```
#### 第三步：解压数据集和转换格式
把数据集下载并解压到 dataset/wider_face/文件夹下,如下层级：
```text
dataset/wider_face/
├── wider_face_split
├── WIDER_test
│   └── images
│       ├── 0--Parade
│       ├── 10--People_Marching
│       ├── 11--Meeting
│       ├── 12--Group
│       ├── 13--Interview
│       ......
├── WIDER_train
│   └── images
│       ├── 0--Parade
│       ├── 10--People_Marching
│       ├── 11--Meeting
│       ├── 12--Group
│       ......
└── WIDER_val
    └── images
        ├── 0--Parade
        ├── 10--People_Marching
        ├── 11--Meeting
        ......
```

```bash
#开始转换为COCO格式并清洗数据集
cd  dataset/wider_face
python widerface2coco.py
#然后会生成 WIDERFaceTrainCOCO.json和WIDERFaceValCOCO.json两个文件。
```
#### 第四步：开始训练
单卡训练：
```bash
python  tools/train.py -c configs/face_detection/ppyface_r50vd_dcn_365e_coco.yml  --use_vdl=true 
```
由于本项目没有使用多卡训练，故不提供相关代码。
如使您想使用自己的数据集以及测试集，需要在配置文件中修改数据集为您自己的数据集。

### 第五步：进行评估
```bash
python tools/eval.py -c configs/face_detection/ppyoloface_r50vd_dcn_365e_coco.yml  -o use_gpu=true
```
由于完整的wider-face还没训练完毕，先放一个先前训练模型,来源[作业三](https://aistudio.baidu.com/aistudio/projectdetail/3521764)评估信息：
```textmate
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=2.52s).
Accumulating evaluation results...
DONE (t=0.07s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.344
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.669
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.329
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.259
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.525
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.077
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.289
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.627
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.733
[02/23 12:10:13] ppdet.engine INFO: Total sample number: 92, averge FPS: 31.507607409541855
```

### 第六步：开始预测

单张图片预测：
```bash
python tools/infer.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml -o \
use_gpu=true weights=output/ppyolov2_r50vd_dcn_365e_coco/model_final.pdparams  \
--infer_img=demo/000000014439.jpg
```
 ![](https://ai-studio-static-online.cdn.bcebos.com/3a13ab9d27124116bff74abb7023b701a5b1e37c31e44494a15a796c0d5cff77)
 
 ![](https://ai-studio-static-online.cdn.bcebos.com/c679c79696b84845a8a7de40d128a157d9f54ab5a4794ffc867a60719c3c5b23)
 
多张图片一起预测：
```bash
python tools/infer.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml -o \
use_gpu=true weights=output/ppyolov2_r50vd_dcn_365e_coco/model_final.pdparams  \
--infer_dir=demo2
```
 ![](https://ai-studio-static-online.cdn.bcebos.com/9173f47cb63f44b49bc8bee07958e261d1924a43365d42e186d889ecadd92248)
 ![](https://ai-studio-static-online.cdn.bcebos.com/d299998408e7408d937b3b27a24ccb65a8ce18a8732644a1b56cb35266836435)



## 4. 其他

关于本项目的详细内容，欢迎访问:[【AI达人创造营第二期】以PP-YOLO v2模型快速实现人脸检测](https://aistudio.baidu.com/aistudio/projectdetail/3525442)
在AI Studio每天都免费硬件资源可以用哟！