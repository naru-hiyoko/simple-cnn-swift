//
//  main.swift
//  simple-cnn-swift
//

import Foundation
import Cocoa

/// for macOS
fileprivate func image2array(_ image: NSImage, scale: Float) -> [Float]
{
    var channel_red = [Float]()
    var channel_blue = [Float]()
    var channel_green = [Float]()
    
    let size = image.size
    
    let bmap : NSBitmapImageRep! = NSBitmapImageRep(data: image.tiffRepresentation!)
    bmap.size = size
    let width = Int(size.width) 
    
    for h in 0..<Int(size.height) {
        for w in 0..<Int(size.width) {
            channel_green.append(Float(bmap.colorAt(x: width - w - 1, y: h)!.greenComponent) * scale)
            channel_red.append(Float(bmap.colorAt(x: width - w - 1, y: h)!.redComponent) * scale)
            channel_blue.append(Float(bmap.colorAt(x: width - w - 1, y: h)!.blueComponent) * scale)
        }
    }
    
    return channel_red + channel_green + channel_blue
    
}

/// main function
public func cnnStartComputing(imagePath: String, netFilePath: String) {
    
    //入力サイズ
    let input_shape = [1, 3, 32, 32]
    
    // 画像を読み込む
    let test_image: NSImage! = NSImage(contentsOfFile: imagePath)
    test_image.size = NSSize(width: 32, height: 32)
    var input_data = image2array(test_image, scale: 255.0)
    
    // ネットワークの読み込み
    let label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "track"]
    let (net, net_params, mean) = load_net(netFilePath)
    
    // 平均画像を入力画像に適用する
    if mean != nil {
        input_data = applyMean(input_data, mean: mean!)
    }
    
    let st = Date()
    let softmax_val : [Float]! = forward(input_data, image_shape: input_shape, net: net, net_params: net_params)
    let end = Date().timeIntervalSince(st)
    print("pred : \(label[softmax_val!.index(of: softmax_val.max()!)!])")
    print("forward time : \(end)")
}


