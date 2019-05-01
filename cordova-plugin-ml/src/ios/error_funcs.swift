//
//  error_funcs.swift
//  NewlarNetwork
//
//  Created by 井上正裕 on 2019/04/30.
//  Copyright © 2019 井上正裕. All rights reserved.
//

import Foundation

//funcを読み出し
var funcs = Logistic()

class SquaredError: Function{
   //se
    func get_value(_: SquaredError, t:Double, y:Double){
        return ((t - y).T @ (t - y)).flatten()[0] / 2.0
    }
    
    func get_derivative(_: SquaredError, t:Double, y:Double){
        return -(t - y)
    }
}


class CrossEntropy: Function{
    //'cross_entropy'

    func get_value(_: CrossEntropy, t:Double, y:Double){
        return -(t.T @ np.log(y)).flatten()[0]
    }

        def get_derivative(_: CrossEntropy, t:Double, y:Double){
        return - t/y
    }
}
