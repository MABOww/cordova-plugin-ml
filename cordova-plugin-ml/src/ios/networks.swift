//
//  networks.swift
//  NewlarNetwork
//
//  Created by 井上正裕 on 2019/04/30.
//  Copyright © 2019 井上正裕. All rights reserved.
//

import Foundation


//import numpy as np

//funcを読み出し
var error_funcs_SquaredError  = SquaredError()
var error_funcs_CrossEntropy  = CrossEntropy()
var error_funcs_CrossEntropy  = CrossEntropy()


from nn.layers import LogisticLayer, TanhLayer, SoftmaxLayer, RectifierLayer




class LayerTypeDoesNotExist(KeyError):
def __init__(self, layer_classes):
self.message = 'Layer type must be %s or %s.' % (
', '.join([cls.name for cls in layer_classes[:-1]]), layer_classes[-1].name)


class ErrorFuncDoesNotExist(KeyError):
def __init__(self, error_funcs):
self.message = 'error_func_name must be %s or %s' % (
', '.join([func.name for func in error_funcs[:-1]]), error_funcs[-1].name)


class Network:
_LAYER_CLASSES = [
LogisticLayer,
TanhLayer,
SoftmaxLayer,
RectifierLayer,
]

_ERROR_FUNCS = [
SquaredError(),
CrossEntropy(),
]

def __init__(self, name, n_input, error_func_name, epsilon):
self._name = name
self._n_input = n_input
self._layers = []
self._error_func = self._get_error_func_by_name(error_func_name)
self._epsilon = epsilon

@property
def name(self):
return self._name

@property
def layers(self):
return self._layers

@property
def epsilon(self):
return self._epsilon

@epsilon.setter
def epsilon(self, value):
self._epsilon = value

@property
def error_func(self):
return self._error_func

@error_func.setter
def error_func(self, name):
self._error_func = self._get_error_func_by_name(name)

def _get_error_func_by_name(self, name):
for func in self._ERROR_FUNCS:
if func.name == name:
return func
    raise ErrorFuncDoesNotExist(self._ERROR_FUNCS)

def add_layer(self, type, n_output, **kwargs):
def get_layer_cls(type):
for cls in self._LAYER_CLASSES:
if cls.name == type:
return cls
raise LayerTypeDoesNotExist(self._LAYER_CLASSES)

n_prev_output = self._layers[-1].n_output if self._layers else self._n_input
layer = get_layer_cls(type)(n_output, n_prev_output, **kwargs)
self._layers.append(layer)

def propagate_forward(self, input_datum):
output = input_datum
for layer in self._layers:
output = layer.propagate_forward(output)
return output

def propagate_backward(self, input_datum, teaching_datum):
delta = self._error_func.get_derivative(teaching_datum, self.propagate_forward(input_datum))

next_layer = None
for layer in reversed(self._layers):
delta = layer.propagate_backward(delta, next_layer.W if next_layer is not None else None)
next_layer = layer

def update(self, input_datum):
prev_layer = None
for layer in self._layers:
layer.update(input_datum if prev_layer is None else prev_layer.y, self._epsilon)
prev_layer = layer

def to_json(self):
return {'meta': {'name': self.name,
    'n_input': self._n_input,
    'error_func': self._error_func.name,
    'epsilon': self.epsilon},
    'layers': [layer.to_json() for layer in self._layers]}

@classmethod
def from_json(cls, json_data):
network = cls(json_data['meta']['name'], json_data['meta']['n_input'],
json_data['meta']['error_func'], json_data['meta']['epsilon'])
for layer in json_data['layers']:
try:
type = layer.pop('type')
except KeyError:
raise LayerTypeDoesNotExist
n_output = len(layer['W'])
network.add_layer(type, n_output, **layer)
return network


class Classifier(Network):
def get_class(self):
return np.argmin(np.abs(1.0 - self._layers[-1].y))
