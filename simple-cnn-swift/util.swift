//
//  util.swift
//  
//
//  Created by 成沢淳史 on 3/23/16.
//
//

import Foundation


func loadJSONFile(_ filename: String) -> Any? {
    print(" ==> loadJSONFile(filename=\(filename)")
    
    //let bundle = NSBundle.mainBundle()
    //let path = bundle.pathForResource(filename, ofType: "json")!
    let path = filename
    guard let jsonData = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
        return nil
    }
    print(" <== loadJSONFile")
    return try? JSONSerialization.jsonObject(with: jsonData, options: .allowFragments)
}


func compute_output_shape(_ height: Int, width: Int, pad_h: Int, pad_w: Int, 
    kernel_h: Int, kernel_w: Int, stride_h: Int, stride_w: Int) -> (Int, Int)
{
    let height_out = Int(ceil( (Float(height) + 2 * Float(pad_h) - Float(kernel_h)) / Float(stride_h) + 1.0 ))
    let width_out = Int(ceil( (Float(width) + 2 * Float(pad_w) - Float(kernel_w)) / Float(stride_w) + 1.0))
    return (width_out, height_out)
}

func copyArrayIF(_ arr : [Int]) -> [Float]
{
    var out = [Float](repeating: 0.0, count: arr.count)
    for i in 0..<out.count {
        out[i] = Float(arr[i])
    }
    return out
}

func copyArrayFF(_ arr : [Float]) -> [Float]
{
    var out = [Float](repeating: 0.0, count: arr.count)
    for i in 0..<out.count {
        out[i] = arr[i]
    }
    return out
}

//平均画像を適用
func applyMean(_ input: [Float], mean: [Float]) -> [Float]
{
    var output: [Float] = [Float](repeating: 0.0, count: input.count)
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

class BatchNorm_params {
    var num_output : Int!
    var E : [Float]!
    var V : [Float]!
    var scale : [Float]!
    var gamma : Float?
    var beta : Float?
    
    init() {
        self.gamma = 1.0
        self.beta = 0.0
    }
    
}


func load_net(_ filename: String) -> (Array<(String, String)>, Dictionary<String, AnyObject>, [Float]?){
    
    let layers_json = loadJSONFile(filename)! as! Dictionary<String, Any>
    
    let layers = layers_json["layer"]! as! Array< Dictionary<String, Any> >
    
    var mean : [Float]?
    if layers_json["mean"] != nil {
        mean = layers_json["mean"] as? [Float]
    }
    
    var net_params : Dictionary<String, AnyObject> = Dictionary()
    var net : Array<(String, String)> = Array()
    
    for i in 0 ..< layers.count {
        let layer = layers[i]
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
        
        if layer_type == "BatchNorm" {
            let blobs = layer["blobs"]! as! [ [String:Any] ]
            let E = blobs[0]["data"]! as! [Float]
            let V = blobs[1]["data"]! as! [Float]
            let scale = blobs[2]["data"] as! [Float]
            
            let cl : BatchNorm_params! = BatchNorm_params()
            cl.E = E
            cl.V = V
            cl.scale = scale
            
            net_params[layer_name] = cl
        }
        
        if layer_type == "Convolution" {
            
            let bottom = (layer["bottom"] as! [String])[0]
            let top = (layer["top"] as! [String])[0]
            
            let blobs = layer["blobs"] as! [[String:Any]]
            let weights = blobs[0]["data"]!
            let bias = blobs[1]["data"]!
            let convolution_param = layer["convolution_param"]! as! [String:Any]
            
            var kernel_size : Int!
            if convolution_param["kernel_size"]! is [Int] {
                kernel_size = (convolution_param["kernel_size"]! as! [Int])[0]
            } else {
                kernel_size = convolution_param["kernel_size"]! as! Int
            }
            
            var pad: Int! = 0
            if convolution_param["pad"]! is [Int] {
                pad = (convolution_param["pad"]! as! [Int])[0]
            } else {
                if convolution_param["pad"] != nil {
                    pad = convolution_param["pad"]! as! Int
                }
            }
            
            var stride: Int! = 1
            if convolution_param["stride"]! is [Int] {
                stride = (convolution_param["stride"]! as! [Int])[0]
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
            cl.weights = weights as! [Float]
            cl.bias = bias as! [Float]
            cl.kernel_size = kernel_size
            cl.pad = pad
            cl.stride = stride
            cl.num_output = num_output
            
            net_params[layer_name] = cl
            
            print(convolution_param)
        }
        
        if layer_type == "Pooling" {
            let bottom = (layer["bottom"]! as! [String])[0]
            let top = (layer["top"]! as! [String])[0]
            
            let pooling_param = layer["pooling_param"]! as! [String:Any]
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

            let inner_product_param = layer["inner_product_param"]! as! [String:Any]
            let num_output : Int = inner_product_param["num_output"]! as! Int
            let top = (layer["top"]! as! [String])[0]
            let bottom = (layer["bottom"]! as! [String])[0]
            let cl : InnerProduct_params! = InnerProduct_params()
            cl.num_output = num_output
            cl.top = top
            cl.bottom = bottom
            net_params[layer_name] = cl
            
            print(inner_product_param)
            
            let blobs = layer["blobs"] as! [[String:Any]]
            
            let weights = blobs[0]["data"] as! [Float]
            let weights_shape = (blobs[0]["shape"]! as! [String:Any])["dim"] as! [Int]
            let bias = blobs[1]["data"] as! [Float]
            cl.weights = weights
            cl.weights_shape = weights_shape
            cl.bias = bias
            
            net_params[layer_name] = cl
            
            print(weights_shape)
        }
        
        if layer_type == "SoftmaxWithLoss" {
            //print(layer)
        }
        
        print("")
        
    }
    
    return (net, net_params, mean)
}

