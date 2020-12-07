# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:torch2onnx.py
# software: PyCharm


from model import Backbone
from config import get_config
import torch
from PIL import Image
import onnx

conf = get_config(False)

model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
model.load_state_dict(torch.load('../work_space/models/model_ir_se50.pth',
                                 map_location=torch.device('cpu')))

dummy_input = torch.randn(1, 3, 112, 112, device=conf.device)

# change to onnx
torch.onnx.export(model=model,
                  args=dummy_input,
                  f='insight.onnx',
                  verbose=True,
                  input_names=['input1'],
                  output_names=['output1'])
