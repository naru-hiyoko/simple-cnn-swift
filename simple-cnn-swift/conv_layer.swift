//
//  conv_layer.swift
//  
//
//  Created by 成沢淳史 on 3/22/16.
//
//

import Foundation
import Metal

class convolution 
{

    /**
     col_dimension : [N, C' , H' , W']

     **/

    static func forward_cpu(_ input: [Float], weights: [Float], bias: [Float], col_dimension: [Int], output_dimension: [Int]) -> [Float]
    {   
        let channels_col = col_dimension[1]
        //let height_col = col_dimension[2]    
        //let width_col = col_dimension[3]
    
        let channels_out = output_dimension[1]
        let height_out = output_dimension[2]
        let width_out = output_dimension[3]
    
        var result : [Float] = [Float](repeating: 0.0, count: Int(channels_out * height_out * width_out))

        let queue = DispatchQueue.global(priority: DispatchQueue.GlobalQueuePriority.high)
        DispatchQueue.concurrentPerform(iterations: Int(channels_out * height_out * width_out), execute: { (id) in 
            let a = (id / (height_out * width_out)) % channels_out
            let b = id % Int(height_out * width_out)
        
            var tmp : Float = bias[a]           
            for c in 0..<channels_col {
                tmp += input[channels_col * b + c] * weights[a * channels_col + c]
            }
            result[height_out * width_out * a + b] = tmp
        })
    
        return result
    }
    
 
}
