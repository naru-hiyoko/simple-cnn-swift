//
//  relu_layer.swift
//  
//
//  Created by 成沢淳史 on 3/23/16.
//
//

import Foundation


func relu_layer(top_data : [Float]) -> [Float] {
    var result = [Float](count: top_data.count, repeatedValue: 0.0)
    for i in 0..<top_data.count {
        result[i] = max(0.0, top_data[i])
    }
    
    return result
}


