//
//  im2col.swift
//  
//
//  Created by 成沢淳史 on 3/22/16.
//
//

import Foundation
import Darwin

/**
called before convolution layer
**/

func col2im(_ data_col: [Float], input_dimension : [Int] , 
    patch_size : Int, stride: Int, pad: Int) -> [Float] {
        let channels : Int = input_dimension[1]
        let height : Int = input_dimension[2]
        let width : Int = input_dimension[3]
        
        let height_col = Int(ceil( (Float(height) + 2 * Float(pad) - Float(patch_size)) / Float(stride) + 1.0 ))
        let width_col = Int(ceil( (Float(width) + 2 * Float(pad) - Float(patch_size)) / Float(stride) + 1.0))
        let channels_col = channels * patch_size * patch_size
        
        var data_im = [Float](repeating: 0.0, count: channels*width*height)
  
        
        for h in 0..<height_col {
            for w in 0..<width_col {
                for c in 0..<channels_col {
                    let c_im : Int = c / patch_size / patch_size
                    let l = c % (patch_size * patch_size)
                    let h_pad = l / patch_size
                    let w_pad = l % patch_size
                    
                    let x = w * stride + w_pad - pad
                    let y = h * stride + h_pad - pad
                    if x >= 0 && x < width && y >= 0 && y < height {
                         data_im[(c_im * height * width) + (y * width) + x] += data_col[(h * width_col + w) * channels_col + c]
                    }
                }
            }
        }
        
        
        return data_im
}

func im2col(_ data_im : [Float], input_dimension: [Int], col_dimension: [Int], kernel_size: Int, stride: Int, pad: Int) -> [Float] {
        

    //let channels: Int = input_dimension[1]
    let height: Int = input_dimension[2]
    let width: Int = input_dimension[3]
        
    let channels_col = col_dimension[1]
    let height_col = col_dimension[2]
    let width_col = col_dimension[3]
    var data_col = [Float](repeating: 0.0, count: channels_col * height_col * width_col) 
    
    for h in 0..<height_col {
        for w in 0..<width_col {
            for c in 0..<channels_col {
                let c_im : Int = c / kernel_size / kernel_size
                let l = c % (kernel_size * kernel_size)
                let h_pad = l / kernel_size
                let w_pad = l % kernel_size
                
                let x = w * stride + w_pad - pad
                let y = h * stride + h_pad - pad
                let id = (width_col * h + w) * channels_col + c
                if x >= 0 && x < width && y >= 0 && y < height {
                    data_col[id] = data_im[(c_im * height * width) + (y * width) + x]
                } else {
                    data_col[id] = 0.0
                }
            }
        }
    }
    
    return data_col
        
}
 




