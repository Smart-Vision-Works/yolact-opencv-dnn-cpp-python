"""
Convert .pth yolact models to onnx

Usage:
    convert_onnx.py <model_file> [options]

options:
    --width=NUM, -w=NUM         the width of an input image. [Default: 256]
    --height=NUM, -h=NUM        the height of an input image. [Default: 256]
    --read_only                 only test readability of model_file.

NOTE: This script is only known to work with pytorch version 1.11.0, and upgrading to 1.12.0 __will__ break it. Please use a proper conda environment.
"""
from docopt import docopt
args = docopt(__doc__)
import torch
import os
import cv2
from yolact import Yolact

if __name__=='__main__':
    if args['--read_only']:
        output_onnx = args['<model_file>']
    else:
        device = 'cpu'
        trained_model = args['<model_file>']
        net = Yolact()
        net.load_weights(trained_model)
        net.eval()
        net.to(device)

        output_onnx = os.path.splitext(trained_model)[0] + '.onnx'
        print(output_onnx)
        print("size ", int(args['--width']), int(args['--height']))
        inputs = torch.randn(1, 3, int(args['--width']), int(args['--height'])).to(device)
        print('convert',output_onnx,'begin')
        torch.onnx.export(net, inputs, output_onnx, verbose=False, opset_version=12, input_names=['image'],
                          output_names=['loc', 'conf', 'mask', 'proto'])
        print('convert', trained_model, 'to onnx finish!!!')

    try:
        dnnnet = cv2.dnn.readNet(output_onnx)
        print('read success')
    except Exception as e:
        print(e)
        print('read failed')
