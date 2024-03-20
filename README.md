# PP-HumanSeg for Sophgo TPU

powerd by [PP-HumanSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/README_cn.md)

Separating human figures and backgrounds at the pixel level is a classic task in image segmentation with a wide range of applications. Generally speaking, this task can be categorized into two types: segmentation targeting half-length portraits, commonly referred to as portrait segmentation; and segmentation targeting both full-body and half-length human figures, commonly known as general human segmentation.

For both portrait and general human segmentation, PaddleSeg has released the PP-HumanSeg series of models, which boast high segmentation accuracy, fast inference speed, and strong versatility. Furthermore, the PP-HumanSeg series models are ready-to-use out of the box, allowing for zero-cost deployment in products, and also support fine-tuning with specific scene data to achieve even better segmentation results.

![test](./pic/human.jpg)
## setup

```bash
pip install -r requirements.txt

# install sail
https://doc.sophgo.com/sdk-docs/v23.10.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html
```

## model & data
you can download my model or export by yourself

### download
```bash
wget https://github.com/wangyifan2018/PP-HumanSeg-TPU/releases/download/v1.0/model.zip
wget https://github.com/wangyifan2018/PP-HumanSeg-TPU/releases/download/v1.0/data.zip

# install if need
sudo apt install unzip

unzip model.zip
unzip data.zip
```

### export model

```bash
# export paddlepaddle -> onnx
pip install paddlepaddle==2.3.0

chmod -R +x scripts
# download model form https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/README_cn.md
./export_onnx.sh

# export onnx -> bmodel
# https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/tpu-mlir/quick_start/html/index.html
./export_bmodel.sh
```

## run

```bash
python src/seg_demo.py \
  --bmodel ./model/pp-humansegv2-mobile_192x192_fp16.bmodel \
  --img_path data/images/human.jpg \
  --save_dir data/images_result/human.jpg \
  --dev_id 5
```
find results in ./data/images_result
