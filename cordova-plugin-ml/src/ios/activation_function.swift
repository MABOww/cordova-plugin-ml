//
//  nn.swift
//  NewlarNetwork
//
//  Created by 井上正裕 on 2019/04/30.
//  Copyright © 2019 井上正裕. All rights reserved.
//

import Foundation

class Logistic: Function{
    //logistic
    func get_value(_: Logistic, s:Double) -> Double{
        return 1 / (1 + np.exp(-s))
    }

    //logistic
    func get_derivative(_: Logistic, y:[Float]) -> [Float]{
        isinstance(y, np.ndarray);
        y.shape[1] == 1
        jacobian = np.zeros(shape=(y.shape[0], y.shape[0]))
        jacobian[np.diag_indices(y.shape[0])] = (y * (1 - y)).flatten()
        return jacobian
    }
}

class Tanh: Function{
    //tanh
    func __init__(_: alpha, _: beta){
        self._alpha = alpha
        self._beta = beta
    }
        
    func alpha(){
        return.self_alpha
    }
        
    func beta(){
        return.self_beta
    }
        
    func get_value(_:test, _: s){// need to change -> test
        return self._alpha * np.tanh(self._beta * s)
    }
     
    func get_derivative(_: test , y:[Float]){// need to change -> test
        isinstance(y, np.ndarray);
        y.shape[1] == 1
        derivative = np.zeros(shape=(y.shape[0], y.shape[0]))
        derivative[np.diag_indices(y.shape[0])] = self._alpha * self._beta * (-np.power(np.tanh(self._beta * y.flatten()), 2))
        return derivative
    }
}


class Softmax: Function{
    // Softmax
    
    func get_value(_: Logistic, s:Double) -> Double {
        _s = s - np.max(s)
        exp_s = np.exp(_s)
        value = exp_s / np.sum(exp_s)
        value[value < 10e-7] = 10e-7
        value[value > 10e+7] = 10e+7
        return value / np.sum(value)
    }
    
    func get_value(_: Logistic, s:Double) -> Double {
        _s = s - np.max(s)
        exp_s = np.exp(_s)
        value = exp_s / np.sum(exp_s)
        value[value < 10e-7] = 10e-7
        value[value > 10e+7] = 10e+7
        return value / np.sum(value)
    }
    
    func get_derivative(_: Logistic, s:Double)-> Double {
        isinstance(y, np.ndarray);
        y.shape[1] == 1
        y_diag = np.zeros(shape=(y.shape[0], y.shape[0]))
        np.fill_diagonal(y_diag, y.flatten())
       // return y_diag - y; @ y.T -> 書き直しが必要
    }
    
}

class Rectifier: Function{
    func get_value(_: Logistic, s:Double)->Double{
        val = deepcopy(s)
        val[val < 0.0] = 0.0
        return val
    }

    func get_derivative(_: Logistic, y:Double)->Double{
        val = deepcopy(y)
        val[val > 0.0] = 1.0
        val[val <= 0.0] = 0.0
        y_diag = np.zeros(shape=(y.shape[0], y.shape[0]))
        np.fill_diagonal(y_diag, val.flatten())
        return y_diag
    }
}

    
    
    
    
    



