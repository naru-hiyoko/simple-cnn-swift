//
//  pooling_layer.swift
//  
//
//  Created by 成沢淳史 on 3/23/16.
//
//

import Foundation

/**
 pool
   MAX: 0
   AVG: 1

**/


enum POOL {
    case max
    case avg
}

func im2colForPool(_ input: [Float], input_shape: [Int], kernel_size: Int, stride: Int, pad: Int) -> [Float] 
{
    
    let a = 2 * Float(pad) - Float(kernel_size)
    let height_col = Int(ceil( (Float(input_shape[2]) + a) / Float(stride) + 1.0 ))
    let width_col = Int(ceil( (Float(input_shape[3]) + a) / Float(stride) + 1.0))
    
    // create output buffer
    let im2colCount = Int(input_shape[1]) * height_col * width_col * kernel_size * kernel_size
    var _in_data_col : [Float] = [Float](repeating: 0.0, count: im2colCount)
    
    
    let k2 = kernel_size * kernel_size
    let cm_offset = k2 * height_col * width_col
    let _in_channel = input_shape[1]
    let _in_height = input_shape[2]
    let _in_width = input_shape[3]
    
    // im2col for pooling layer
    for n in 0..<_in_channel {
        for h in 0..<height_col {
            for w in 0..<width_col {
                for c in 0..<kernel_size * kernel_size {   
                    let h_pad : Int = c / kernel_size
                    let w_pad : Int = c % kernel_size
                    let x : Int = w * stride + w_pad - pad
                    let y : Int = h * stride + h_pad - pad 
                    let id = cm_offset * n + (k2 * width_col) * h + k2 * w + c
                    if x >= 0 && x < _in_width && y >= 0 && y < _in_height {
                        _in_data_col[id] = input[(_in_height * _in_width) * n +  _in_width * y + x]
                    } else {
                        _in_data_col[id] = -1000.0
                    }
                    
                }
            }
        }
    }
    
    return _in_data_col
    
}


func max_pool(_ input: [Float], input_shape: [Int], kernel_size : Int,  stride : Int, pad: Int) -> [Float]
{
    
    let input_data = im2colForPool(input, input_shape: input_shape, kernel_size: kernel_size, stride: stride, pad: pad)
    
    let channels : Int = input_shape[1]
    let height : Int = input_shape[2]
    let width : Int = input_shape[3]
    
    let height_col = Int(ceil( (Float(height) + 2 * Float(pad) - Float(kernel_size)) / Float(stride) + 1.0 ))
    let width_col = Int(ceil( (Float(width) + 2 * Float(pad) - Float(kernel_size)) / Float(stride) + 1.0))
    
    var result = [Float](repeating: 0.0, count: channels * height_col * width_col)
    
    let k2 = kernel_size * kernel_size
    
    let queue = DispatchQueue.global(priority: DispatchQueue.GlobalQueuePriority.high)
    
    DispatchQueue.concurrentPerform(iterations: channels, execute: { (id) in 
        
        let c_im = width_col * height_col * id
        
        for n in 0..<width_col * height_col {
            var tmp_result : Float = -1000.0
            
            for c in 0..<kernel_size*kernel_size {
                let valAt = input_data[(k2 * width_col * height_col * id) + k2 * n + c]
                tmp_result = max(tmp_result, valAt)
                
            }
            result[c_im + n] = tmp_result
        }
        
    })
    
    
    return result
}


func avg_pool(_ input: [Float], input_shape: [Int], kernel_size : Int, stride : Int, pad : Int) -> [Float]
{

    
    let channels : Int = input_shape[1]
    let height : Int = input_shape[2]
    let width : Int = input_shape[3]
    
    let input_data = im2colForPool(input, input_shape: input_shape, kernel_size: kernel_size, stride: stride, pad: pad)    
    
    let height_col = Int(ceil( (Float(height) + 2 * Float(pad) - Float(kernel_size)) / Float(stride) + 1.0 ))
    let width_col = Int(ceil( (Float(width) + 2 * Float(pad) - Float(kernel_size)) / Float(stride) + 1.0))
    
    var result = [Float](repeating: 0.0, count: channels * height_col * width_col)
    
    let k2 = kernel_size * kernel_size
    
    let queue = DispatchQueue.global(priority: DispatchQueue.GlobalQueuePriority.high)
    
    DispatchQueue.concurrentPerform(iterations: channels, execute: { (id) in
        let c_im = width_col * height_col * id
        
        for n in 0..<width_col * height_col {
            var tmp_result : Float = 0.0
            var dummy_size : Int = 0
            
            for c in 0..<k2 {
                let valAt = input_data[(k2 * width_col * height_col * id) + k2 * n + c]
                if valAt != -1000.0 {
                    dummy_size += 1
                    tmp_result += valAt
                }
            }
            
            result[c_im + n] = tmp_result / Float(dummy_size)
        }
    })
       
    return result
}

