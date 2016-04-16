import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from google.protobuf.descriptor import FieldDescriptor as FD
import json
import sys
import os
import numpy as np
import argparse

def load_meanfile(file):
    print 'loaded mean file.'
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(file, 'rb').read()
    blob.ParseFromString(data)
    mean = np.array(caffe.io.blobproto_to_array(blob))[0]
    return mean

debug = False

def pb2json(pb):
    _ftype2js = {
        FD.TYPE_DOUBLE: float,
        FD.TYPE_FLOAT: float,
        FD.TYPE_INT64: long,
        FD.TYPE_UINT64: long,
        FD.TYPE_INT32: int,
        FD.TYPE_FIXED64: float,
        FD.TYPE_FIXED32: float,
        FD.TYPE_BOOL: bool,
        FD.TYPE_STRING: unicode,
        FD.TYPE_BYTES: lambda x: x.encode('string_escape'),
        FD.TYPE_UINT32: int,
        FD.TYPE_ENUM: int,
        FD.TYPE_SFIXED32: float,
        FD.TYPE_SFIXED64: float,
        FD.TYPE_SINT32: int,
        FD.TYPE_SINT64: long,
    }
    js = {}
    fields = pb.ListFields()
    for field, value in fields:
        if field.type == FD.TYPE_MESSAGE:
            ftype = pb2json
        elif field.type in _ftype2js:
            ftype = _ftype2js[field.type]
        else:
            print("WARNING: Field %s.%s of type '%d' is not supported" % (pb.__class__.__name__, field.name, field.type, ))
        if field.label == FD.LABEL_REPEATED:
            js_value = []
            for v in value:
                js_value.append(ftype(v))
            if debug:
                if len(js_value) > 64 or (field.name == 'data' and len(js_value) > 8):
                    head_n = 5
                    js_value = js_value[:head_n] + ['(%d elements more)' % (len(js_value) - head_n)]
        else:
            js_value = ftype(value)
        js[field.name] = js_value
    return js


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        help = '*.caffemodel'
    )

    parser.add_argument(
        '--mean',
        help = 'meanfile : *.binaryproto'
    )
    
    parser.add_argument(
        '--debug',
        action = 'store_true',
        help = 'write only top 5 weight param.'
    )
    args = parser.parse_args()
    debug = args.debug
    
    net = caffe_pb2.NetParameter()
    caffemodel = args.model
    assert '.caffemodel' in caffemodel, 'invalid input file'
    print 'loading network parameters from {}'.format(caffemodel)
    with open(caffemodel) as f:
        net.ParseFromString(f.read())

    print 'saving... {}.json'.format(os.path.basename(caffemodel).split('.')[0])
    js = pb2json(net)

    """ load mean file """
    if args.mean:
        mean = load_meanfile(args.mean)
        mean_arr = []
        for c in range(mean.shape[0]):
            for h in range(mean.shape[1]):
                for w in range(mean.shape[2]):
                    mean_arr.append(mean[c, h, w])
        js['mean'] = mean_arr

    
    f = open('{}.json'.format(os.path.basename(caffemodel).split('.')[0]), 'w')
    json.dump(js, f, indent=2)

    print 'done'
