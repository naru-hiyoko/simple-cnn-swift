//
//  inner_product_layer.swift
//  dnn
//
//  Created by 成沢淳史 on 3/28/16.
//  Copyright © 2016 成沢淳史. All rights reserved.
//

import Foundation


func inner_product(input: [Float], weights: [Float], weights_shape: [Int], bias: [Float]) -> [Float] 
{
    var output: [Float] = [Float](count: weights_shape[0], repeatedValue: 0.0)
    let num_output = weights_shape[0]
    
    let queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0)
    
    
    dispatch_apply(num_output, queue, { (c) in 
        output[c] = bias[c]
        for i in 0..<weights_shape[1] {
            output[c] += input[i] * weights[weights_shape[1] * c + i]
        }
    })
    
    
    return output
}



// weights_shape[0] .. n of top data
// weights_shape[1] .. n of bottom data

func inner_product_backward(input: [Float], weights: [Float], weights_shape: [Int]) -> [Float]
{
    var output: [Float] = [Float](count: weights_shape[1], repeatedValue: 0.0)
    for c in 0..<weights_shape[0] {
        for i in 0..<weights_shape[1] {
            output[i] += input[c] * weights[weights_shape[1] * c + i]
        }
    }
    
    
    return output
}



