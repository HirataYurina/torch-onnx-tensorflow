# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:onnx2pb.py
# software: PyCharm

import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

'''
    if system and package version are not compatible,
    can not exchange successfully!!
    
    i use:
    python=3.7
    onnx=1.8
    onnx_tf=1.7
    tf=2.2.0
    linux
'''

model = onnx.load('./insight.onnx')
tf_model = prepare(model)
tf_model.export_graph('./insight.pb')
