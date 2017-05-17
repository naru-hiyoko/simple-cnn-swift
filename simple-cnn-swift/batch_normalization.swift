//
//  batch_normalization.swift
//  simple-cnn-swift
//
//  Created by 成沢淳史 on 6/16/16.
//  Copyright © 2016 成沢淳史. All rights reserved.
//

import Foundation

func batch_normalization(_ _in : [Float], _in_dimension : [Int], E : [Float], V : [Float], scale : [Float], 
    gamma : Float = 1.0, beta : Float = 0.0)-> [Float]
{
    let num_output : Int = _in_dimension[1] 
    let height : Int = _in_dimension[2]
    let width : Int = _in_dimension[3]
    
    var result : [Float] = [Float](repeating: 0.0, count: num_output * height * width)
    
    for n in 0..<num_output {
        for h in 0..<height {
            for w in 0..<width {
                let id : Int = (height * width) * n + width * h + w
                let _e = E[n] / scale[0]
                let _v = V[n] / scale[0]
                let tmp = _in[id] * gamma / sqrt(_v + 0.00001) + (beta - gamma * _e / sqrt(_v + 0.00001))
                result[id] = tmp
                
            }
        }
    }
    return result
    
}
