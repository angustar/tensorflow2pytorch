import torch
import tensorflow as tf
import deepdish as dd
import argparse
import numpy as np
import pretrainedmodels


def tr(v):
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3, 2, 0, 1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v


def read_ckpt(ckpt):
    reader = tf.compat.v1.train.NewCheckpointReader(ckpt)
    weights = {n: reader.get_tensor(n) for (
        n, _) in reader.get_variable_to_shape_map().items()}
    pyweights = {k: tr(v) for (k, v) in weights.items()}
    return pyweights


def main(args):
    model_name = args.model
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

    model_dict = model.state_dict()
    new_pre_dict = {}
    weights = read_ckpt(args.input)
    for k, v in weights.items():
        new_pre_dict[k] = torch.Tensor(v)
    model_dict.update(new_pre_dict)
    model.load_state_dict(model_dict)
    
    torch.save(model, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='inceptionv3',
                        help="https://github.com/Cadene/pretrained-models.pytorch#available-models")
    parser.add_argument("--input", type=str, default='models/ckpt/ens4_adv_inception_v3_rename.ckpt.index',
                        help="path of the input '.index' file")
    parser.add_argument("--output", type=str, default='models/pth/ens4_adv_inception_v3_rename.pth',
                        help="path of the output '.pth' file")

    args = parser.parse_args()

    main(args)
