//
//  layers.swift
//  NewlarNetwork
//
//  Created by 井上正裕 on 2019/04/30.
//  Copyright © 2019 井上正裕. All rights reserved.
//

import Foundation

//import numpy as np -> 代替の行列演算用関数が必要
//funcを読み出し
var activaton_function_Logistic  = Logistic()
var activaton_function_Tanh  = Tanh()
var activaton_function_Softmax  = Softmax()
var activaton_function_Rectifier  = Rectifier()


class BaseLayer{
 // base

    func __init__(_: BaseLayer, n_output:Double, n_prev_output:Double, f:Double){
        """
        :param n_output: Number of this layer's output
        :param n_prev_output: Number of previous layer's output
        :param f: Activation function (callable)
        :param df: Derivative of the activation function (callable)
        """
        self._W = self._init_W(n_output, n_prev_output)
        self._b = self._init_b(n_output)
        self._f = f
        self._y = None
        self._delta = None
        }

    func __str__(_: BaseLayer){
        return self.name
    }

    func _init_W(_: BaseLayer, n_output:Double, n_prev_output:Double, kwargs: Double...){
        return np.random.uniform(-1, 1, size=(n_output, n_prev_output))
    }

    func _init_b(_: BaseLayer, n_output:Double, kwargs: Float...){
        return np.random.uniform(-1, 1, size=(n_output, 1))
    }

//property
    func n_output(_: BaseLayer){
        return self._W.shape[0]
    }

//property
    func W(_: BaseLayer){
        return self._W
    }

//W.setter
    func W(_: BaseLayer, value:Double){
        return self._W
    }
    
//property
    func b(_: BaseLayer){
        return self._b
    }

//b.setter
    func b(_: BaseLayer, value:Double){
        self._b = value
    }

//property
    func ave_abs_W(_: BaseLayer){
        return np.average(np.abs(self._W))
    }

//property
    func ave_W(_: BaseLayer){
        return np.average(self._W)
    }

//property
    func y(_: BaseLayer){
        return self._y
    }

//property
    func delta(_: BaseLayer){
        return self._delta
    }

    func propagate_forward(_: BaseLayer, x:Double){
        self._y = self._f.get_value(self._W @ x + self._b)
        return self._y
    }

    func propagate_backward(_: BaseLayer, next_delta:Double, next_W:Double){
        if next_W is not None{
            self._delta = self._f.get_derivative(self._y) @ next_W.T @ next_delta
        }else{
            self._delta = self._f.get_derivative(self._y) @ next_delta
        }
        return self._delta
    }


func update(self, prev_y, epsilon)
Delta_W = self._delta @ prev_y.T
self._W -= epsilon * Delta_W
self._b -= epsilon * self._delta

def to_json(self):
return {'type': self.name, 'W': self._W.tolist(), 'b': self._b.tolist()}


class LogisticLayer(BaseLayer):
name = 'logistic'

def __init__(self, n_output, n_prev_output):
super().__init__(n_output, n_prev_output, Logistic())


class TanhLayer(BaseLayer):
name = 'tanh'

def __init__(self, n_output, n_prev_output, alpha, beta):
super().__init__(n_output, n_prev_output, Tanh(alpha, beta))

def to_json(self):
data = super(TanhLayer, self).to_json()
data['alpha'] = self._f.alpha
data['beta'] = self._f.beta
return data
}

class SoftmaxLayer(BaseLayer):
name = 'softmax'

def __init__(self, n_output, n_prev_output):
super().__init__(n_output, n_prev_output, Softmax())


class RectifierLayer(BaseLayer):
name = 'rectifier'

def __init__(self, n_output, n_prev_output):
super().__init__(n_output, n_prev_output, Rectifier())
