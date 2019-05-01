//
//  func.swift
//  NewlarNetwork
//
//  Created by 井上正裕 on 2019/04/30.
//  Copyright © 2019 井上正裕. All rights reserved.
//

import Foundation



class Function{
    //name
    
    func get_value(_: Function, args: Float, kwargs: Float){
        return raise(NotImplementedError)
    }
    
    func get_derivative(_: Function, args: Float, kwargs: Float){
        return raise(NotImplementedError)
    }
    
}


