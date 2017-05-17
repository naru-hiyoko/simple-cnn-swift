//
//  relu_layer.swift
//  
//
//  Created by 成沢淳史 on 3/23/16.
//
//

import Foundation


func relu_layer(_ top_data : [Float]) -> [Float] {
    var result = [Float](repeating: 0.0, count: top_data.count)
    for i in 0..<top_data.count {
        result[i] = max(0.0, top_data[i])
    }
    
    return result
}


