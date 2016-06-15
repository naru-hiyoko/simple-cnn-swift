//
//  classify.swift
//  dnn
//
//  Created by 成沢淳史 on 3/24/16.
//  Copyright © 2016 成沢淳史. All rights reserved.
//

import Foundation
import Cocoa
import Metal

func forward(image : [Float], image_shape : [Int], net : Array<(String, String)>, net_params : Dictionary<String, AnyObject>) -> [Float]?
{
    // [N, C, H, W]
    var input_data : [Float]!
    var input_shape : [Int]!
    
    //var output_data : [Float]!
    //var output_shape : [Int]!
    
    
    for (name, type) in net {
        
        print("\(name) : \(type)")
        
        if type == "Data" {
     
            input_data = image
            input_shape = image_shape
       
            
        }
        
        if type == "Convolution" {
            print("<-- \(input_shape)")
            
            let params : Conv_params! = net_params[name] as! Conv_params
            let weights = params.weights
            let bias = params.bias
            let kernel_size = params.kernel_size
            let pad = params.pad
            let stride = params.stride
            let num_output = params.num_output
            
            let channels_col = Int(input_shape[1]) * kernel_size * kernel_size
            let height_col = Int(ceil( (Float(input_shape[2]) + 2 * Float(pad) - Float(kernel_size)) / Float(stride) + 1.0 ))
            let width_col = Int(ceil( (Float(input_shape[3]) + 2 * Float(pad) - Float(kernel_size)) / Float(stride) + 1.0))
            
            let col_dimension : [Int] = [1, channels_col, height_col, width_col]
            
            // do im2col
            input_data = im2col(input_data, input_dimension: input_shape, col_dimension: col_dimension, kernel_size: kernel_size,stride: stride, pad: pad)

            //let im2ColCount = channels_col * height_col * width_col
            var output_dimension : [Int] = [1, num_output, height_col, width_col]
            let outputCount = output_dimension[1] * output_dimension[2] * output_dimension[3]          
            var output = [Float](count: outputCount, repeatedValue: 0.0)
 
            // compute
            output = convolution.forward_cpu(input_data, weights: weights, bias: bias, col_dimension: col_dimension, output_dimension: output_dimension)
            
            input_data = output
            input_shape = output_dimension
            
            print("--> \(output_dimension)")
        }
        
        if type == "Pooling" {
            print("<-- \(input_shape)")
            
            let params : Pool_params! = net_params[name] as! Pool_params
            let kernel_size = params.kernel_size
            let stride = params.stride
            let pad = params.pad
            let pool = params.pool
            
            let height_col = Int(ceil( (Float(input_shape[2]) + 2 * Float(pad) - Float(kernel_size)) / Float(stride) + 1.0 ))
            let width_col = Int(ceil( (Float(input_shape[3]) + 2 * Float(pad) - Float(kernel_size)) / Float(stride) + 1.0))
            
            let output_shape : [Int] = [1, input_shape[1], height_col, width_col]
            var output : [Float] = [Float](count: input_shape[1] * height_col * width_col, repeatedValue: 0.0)
            
            if pool == 0 {
                output = max_pool(input_data, input_shape: input_shape, kernel_size: kernel_size, stride: stride, pad: pad)
            } 
            
            if pool == 1 {
                output = avg_pool(input_data, input_shape: input_shape, kernel_size: kernel_size, stride: stride, pad: pad)
            }
            
            input_data = output
            input_shape = output_shape
            
            print("--> \(input_shape)")
            
        }
        
        if type == "ReLU" {
            print("<-- \(input_shape)")
            let output: [Float] = relu_layer(input_data)
            input_data = output
            print("--> \(input_shape)")
            
        }
        
        
        if type == "InnerProduct" {
            print("<-- \(input_shape)")
            let params : InnerProduct_params! = net_params[name] as! InnerProduct_params
            let weights : [Float]! = params.weights
            let weights_shape : [Int]! = params.weights_shape
            let bias : [Float] = params.bias
            let num_output = params.num_output
            
            var output : [Float] = [Float](count: num_output, repeatedValue: 0.0)
            let output_shape : [Int]! = [1, num_output, 1, 1]
            
            output = inner_product(input_data, weights: weights, weights_shape: weights_shape, bias: bias)

            
            input_data = output
            input_shape = output_shape
            
            print("--> \(input_shape)")
        
        }
        
        if type == "BatchNorm" {
            let params : BatchNorm_params! = net_params[name] as! BatchNorm_params
            let E : [Float]! = params.E
            let V : [Float]! = params.V
            let scale : [Float]! = params.scale

            var output : [Float]!
            output = batch_normalization(input_data, _in_dimension: input_shape, E: E, V: V, scale: scale)
            
            print("<-- \(input_shape)")
            input_data = output
            print("--> \(input_shape)")
        
        }
        
        if type == "SoftmaxWithLoss" {
            print("<-- \(input_shape)")
            print(input_data)
            // Soft-max function
            let t = input_data.map { exp($0) }
            let total = t.reduce(0, combine: +)
            let result = t.map { $0 / total }
            print(result)
            let maxVal : Float! = result.maxElement()
            print("MaxVal : \(maxVal)")
            print("index : \(result.indexOf(maxVal)!)")
            return result
        }
        
    }
    
    return nil
}

