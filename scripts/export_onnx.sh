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
