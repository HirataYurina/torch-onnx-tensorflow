# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:onnx_pred.py
# software: PyCharm

import onnx
import onnxruntime
from config import get_config
import torch
from PIL import Image

conf = get_config(False)

model = onnx.load('./insight.onnx')

print(onnx.checker.check_model(model))

# print(onnx.helper.printable_graph(model.graph))

face1 = Image.open('test_img/test1_res.jpg')
face2 = Image.open('test_img/test2_res.jpg')

# use onnxruntime to infer
session = onnxruntime.InferenceSession('./insight.onnx')
inputs_name = session.get_inputs()[0].name
print(inputs_name)
outputs_name = session.get_outputs()[0].name
print(outputs_name)

embed1 = session.run([], {'input1': conf.test_transform(face1).to('cpu').unsqueeze(0).numpy()})
# print(embed1)
embed2 = session.run([], {'input1': conf.test_transform(face2).to('cpu').unsqueeze(0).numpy()})

distance = torch.sum(torch.pow(torch.tensor(embed1[0] - embed2[0]), 2))
print(distance)
