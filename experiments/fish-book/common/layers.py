"""层类汇总 —— 从 ch5 目录导入，提供与鱼书一致的别名。"""
import importlib

_affine = importlib.import_module("affineLayer")
_relu = importlib.import_module("reluLayer")
_sigmoid = importlib.import_module("sigmoidLayer")
_add = importlib.import_module("addLayer")
_multi = importlib.import_module("multiLayer")
_softmax = importlib.import_module("softmaxLayer")

Affine = _affine.AffineLayer
Relu = _relu.ReLULayer
Sigmoid = _sigmoid.SigmoidLayer
AddLayer = _add.AddLayer
MultiLayer = _multi.MultiLayer
SoftmaxWithLoss = _softmax.SoftmaxWithLoss
Softmax = _softmax.Softmax
