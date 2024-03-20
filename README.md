# PP-HumanSeg for Sophgo TPU

powerd by [PP-HumanSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/README_cn.md)

Separating human figures and backgrounds at the pixel level is a classic task in image segmentation with a wide range of applications. Generally speaking, this task can be categorized into two types: segmentation targeting half-length portraits, commonly referred to as portrait segmentation; and segmentation targeting both full-body and half-length human figures, commonly known as general human segmentation.

For both portrait and general human segmentation, PaddleSeg has released the PP-HumanSeg series of models, which boast high segmentation accuracy, fast inference speed, and strong versatility. Furthermore, the PP-HumanSeg series models are ready-to-use out of the box, allowing for zero-cost deployment in products, and also support fine-tuning with specific scene data to achieve even better segmentation results.

## setup

```bash
pip install -r requirements.txt

# install sail
https://doc.sophgo.com/sdk-docs/v23.10.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html
```

## run

```bash
python src/seg_demo.py \
  --bmodel ./model/pp-humansegv2-mobile_192x192_fp32.bmodel \
  --img_path data/images/portrait_heng.jpg \
  --save_dir data/images_result/portrait_heng_v2.jpg \
  --dev_id 5
```

## export model

```bash
pip install paddlepaddle==2.3.0

# download model form https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/README_cn.md
python paddle_infer_shape.py  --model_dir . \
                          --model_filename model.pdmodel \
                          --params_filename model.pdiparams \
                          --save_dir human_pp_humansegv2_mobile_192x192 \
                          --input_shape_dict="{'x':[1,3,192,192]}"

pip install paddle2onnx
paddle2onnx  --model_dir ./human_pp_humansegv2_mobile_192x192 \
          --model_filename model.pdmodel \
          --params_filename model.pdiparams \
          --opset_version 13 \
          --save_file human_pp_humansegv2_mobile_192x192.onnx

```