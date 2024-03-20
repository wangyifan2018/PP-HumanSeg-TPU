model_transform.py \
  --model_name pp-humansegv2-mobile_192x192 \
  --model_def human_pp_humansegv2_mobile_192x192.onnx \
  --input_shapes [[1,3,192,192]] \
  --mean 0.0,0.0,0.0 \
  --scale 0.0039216,0.0039216,0.0039216 \
  --mlir pp-humansegv2-mobile_192x192.mlir

model_deploy.py \
  --mlir pp-humansegv2-mobile_192x192.mlir \
  --quantize F16 \
  --chip bm1684x \
  --model pp-humansegv2-mobile_192x192_fp16.bmodel \
  --compare_all \
  --debug

model_deploy.py \
  --mlir pp-humansegv2-mobile_192x192.mlir \
  --quantize F32 \
  --chip bm1684x \
  --model pp-humansegv2-mobile_192x192_fp32.bmodel \
  --compare_all \
  --debug