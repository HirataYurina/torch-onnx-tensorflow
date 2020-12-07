# How to Use ONNX

<img src="./onnx.jpg">

[![Build Status](https://travis-ci.org/onnx/onnx-tensorflow.svg?branch=master)](https://travis-ci.org/onnx/onnx-tensorflow)

### 1.pip install

```shell
pip install onnx

pip install onnx_tf

pip install onnxruntime
```

**You need to pay attention to version of onnx and tensorflow.**

**The last onnx_tf is supported for tensorflow >= 2.2.0.**



### 2.torch to onnx

```pyth
import torch

torch.onnx.export(model=,
				  args=,
				  f=,
				  input_names=,
				  output_names=)
```

### 3.onnx to tensorflow

```pyth
import onnx
import onnx_tf

model = onnx.load(file_path)
tf_export = onnx_tf.prepare(model)
tf_export.export_graph('XXX.pb')
```

### 4.use onnx to infer

```pyth
import onnxruntime

sess = onnxruntime.InferenceSession('XXX.onnx')
inputs_name = sess.get_inputs[0].name
outputs_name = sess.get_outputs[0].name

results = sess.run([], {'input1': [array format]})
```



