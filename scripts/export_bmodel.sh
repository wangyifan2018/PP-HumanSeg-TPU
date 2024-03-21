model_transform.py \
  --model_name pp-humansegv2-mobile_192x192 \
  --model_def human_pp_humansegv2_mobile_192x192.onnx \
  --input_shapes [[1,3,192,192]] \
  --mean 0.0,0.0,0.0 \
  --scale 0.0039216,0.0039216,0.0039216 \
  --mlir pp-humansegv2-mobile_192x192.mlir

# fp16
model_deploy.py \
  --mlir pp-humansegv2-mobile_192x192.mlir \
  --quantize F16 \
  --chip bm1684x \
  --model pp-humansegv2-mobile_192x192_fp16.bmodel \
  --compare_all \
  --debug

# fp32
model_deploy.py \
  --mlir pp-humansegv2-mobile_192x192.mlir \
  --quantize F32 \
  --chip bm1684x \
  --model pp-humansegv2-mobile_192x192_fp32.bmodel \
  --compare_all \
  --debug

# int8
run_calibration.py pp-humansegv2-mobile_192x192.mlir \
  --dataset ./data/mini_supervisely/Images \
  --input_num 300 \
  -o pp-humansegv2-mobile_192x192_cali_table


model_deploy.py \
  --mlir pp-humansegv2-mobile_192x192.mlir \
  --quantize INT8 \
  --calibration_table pp-humansegv2-mobile_192x192_cali_table \
  --chip bm1684x \
  --model pp-humansegv2-mobile_192x192_int8.bmodel \
  --test_input pp-humansegv2-mobile_192x192_in_f32.npz \
  --test_reference pp-humansegv2-mobile_192x192.npz \
  --compare_all \
  --debug