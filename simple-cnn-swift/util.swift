//
//  util.swift
//  
//
//  Created by 成沢淳史 on 3/23/16.
//
//

import Foundation


func loadJSONFile(filename: String) -> NSDictionary? {
    print(" ==> loadJSONFile(filename=\(filename)")
    
    do {
        //let bundle = NSBundle.mainBundle()
        //let path = bundle.pathForResource(filename, ofType: "json")!
        let path = filename
        let jsonData = NSData(contentsOfFile: path)
        print(" <== loadJSONFile")    
        return try NSJSONSerialization.JSONObjectWithData(jsonData!, options: .AllowFragments) as? NSDictionary
    } catch _ {
        return nil
    }
    
}


func compute_output_shape(height: Int, width: Int, pad_h: Int, pad_w: Int, 
    kernel_h: Int, kernel_w: Int, stride_h: Int, stride_w: Int) -> (Int, Int)
{
    let height_out = Int(ceil( (Float(height) + 2 * Float(pad_h) - Float(kernel_h)) / Float(stride_h) + 1.0 ))
    let width_out = Int(ceil( (Float(width) + 2 * Float(pad_w) - Float(kernel_w)) / Float(stride_w) + 1.0))
    return (width_out, height_out)
}

func copyArrayIF(arr : [Int]) -> [Float]
{
    var out = [Float](count: arr.count, repeatedValue: 0.0)
    for i in 0..<out.count {
        out[i] = Float(arr[i])
    }
    return out
}

func copyArrayFF(arr : [Float]) -> [Float]
{
    var out = [Float](count: arr.count, repeatedValue: 0.0)
    for i in 0..<out.count {
        out[i] = arr[i]
    }
    return out
}

//平均画像を適用
func applyMean(input: [Float], mean: [Float]) -> [Float]
{
    var output: [Float] = [Float](count: input.count, repeatedValue: 0.0)
    for i in 0..<input.count {
        output[i] = input[i] - mean[i]
    }
    return output
}



class Conv_params {
    var bottom : String!
    var top : String!
    var weights : [Float]!
    var bias : [Float]!
    var kernel_size : Int!
    var pad: Int!
    var stride: Int!
    var num_output: Int!
    
}

class Pool_params {
    var bottom : String!
    var top : String!
    var kernel_size : Int!
    var pool : Int!
    var pad: Int!
    var stride: Int!
}

class InnerProduct_params {
    var num_output: Int!
    var top: String!
    var bottom : String!
    var weights : [Float]!
    var weights_shape : [Int]!
    var bias : [Float]!
}


func load_net(filename: String) -> (Array<(String, String)>, Dictionary<String, AnyObject>, [Float]?){
    
    let layers_json = loadJSONFile(filename)
    let layers = layers_json!["layer"]
    
    var mean : [Float]?
    if layers_json!["mean"] != nil {
        mean = layers_json!["mean"] as? [Float]
    }
    
    var net_params : Dictionary<String, AnyObject> = Dictionary()
    var net : Array<(String, String)> = Array()
    
    for i in 0..<layers!.count {
        let layer = layers![i] as! NSDictionary
        let layer_name = layer["name"]! as! String
        let layer_type = layer["type"]! as! String
        
        
        
        print("\(layer_name): \(layer_type)")
        net.append((layer_name, layer_type))
        
        if layer_type == "Data" {
            //print(layer)
        }
        
        if layer_type == "Dropout" {
            //print(layer)
        }
        
        if layer_type == "Convolution" {
            
            let bottom = layer["bottom"]![0] as! String
            let top = layer["top"]![0] as! String
            
            
            let blobs = layer["blobs"]!        
            let weights : [Float] = blobs[0]["data"]! as! [Float]
            let bias : [Float] = blobs[1]["data"]! as! [Float]
            let convolution_param = layer["convolution_param"]! as! NSDictionary
            
            var kernel_size : Int!
            if convolution_param["kernel_size"]?[0] != nil {
                kernel_size = convolution_param["kernel_size"]![0] as! Int
            } else {
                kernel_size = convolution_param["kernel_size"]! as! Int
            }
            
            var pad: Int! = 0
            if convolution_param["pad"]?[0] != nil {
                pad = convolution_param["pad"]![0] as! Int
            } else {
                if convolution_param["pad"] != nil {
                    pad = convolution_param["pad"]! as! Int
                }
            }
            
            var stride: Int! = 1
            if convolution_param["stride"]?[0] != nil {
                stride = convolution_param["stride"]![0] as! Int
            } else {
                if convolution_param["stride"] != nil {
                    stride = convolution_param["stride"]! as! Int
                }
            }
            
            let num_output = convolution_param["num_output"]! as! Int
            
            //print(blobs[0]?["shape"])
            
            
            let cl : Conv_params! = Conv_params()
            cl.bottom = bottom
            cl.top = top
            cl.weights = weights
            cl.bias = bias
            cl.kernel_size = kernel_size
            cl.pad = pad
            cl.stride = stride
            cl.num_output = num_output
            
            net_params[layer_name] = cl
            
            print(convolution_param)
        }
        
        if layer_type == "Pooling" {
            let bottom = layer["bottom"]![0] as! String
            let top = layer["top"]![0] as! String
            
            
            let pooling_param = layer["pooling_param"]! as! NSDictionary
            let kernel_size = pooling_param["kernel_size"] as! Int
            let pool = pooling_param["pool"] as! Int
            let stride = pooling_param["stride"]! as! Int
            
            var pad : Int! = 0
            if pooling_param["pad"] != nil {
                pad = pooling_param["pad"]! as! Int
            }
            
            let cl : Pool_params! = Pool_params()
            cl.top = top
            cl.bottom = bottom
            cl.kernel_size = kernel_size
            cl.pad = pad
            cl.pool = pool
            cl.stride = stride
            
            net_params[layer_name] = cl
            
            print(pooling_param)
            
        }
        
        if layer_type == "InnerProduct" {

            let inner_product_param = layer["inner_product_param"]! as! NSDictionary
            let num_output : Int = inner_product_param["num_output"]! as! Int
            let top = layer["top"]![0] as! String
            let bottom = layer["bottom"]![0] as! String
            let cl : InnerProduct_params! = InnerProduct_params()
            cl.num_output = num_output
            cl.top = top
            cl.bottom = bottom
            net_params[layer_name] = cl
            
            print(inner_product_param)
            
            let weights = layer["blobs"]![0]["data"] as! [Float]
            let weights_shape = layer["blobs"]![0]!["shape"]!!["dim"] as! [Int]
            let bias = layer["blobs"]![1]["data"] as! [Float]
            cl.weights = weights
            cl.weights_shape = weights_shape
            cl.bias = bias
            
            net_params[layer_name] = cl
            
            print(weights_shape)
            //print(weights.count)
            //print(layer["blobs"]?[1]?["shape"])
            //print(bias.count)
            
        }
        
        if layer_type == "SoftmaxWithLoss" {
            //print(layer)
        }
        
        
        print("")
        
    }
    
    return (net, net_params, mean)
    
    
    
}