import argparse
import re
from typing import Dict

import torch
from hydra import compose, initialize
from keras import Model, ops
from numpy import ndarray
from omegaconf import OmegaConf
from torch import Tensor
from kyolo.model.build import build_model

VAR_NAME_MAP = {
    "conv.weight": "conv/kernel",
    "conv.bias": "conv/bias",
    "bn.weight": "bn/gamma",
    "bn.bias": "bn/beta",
    "bn.running_mean": "bn/moving_mean",
    "bn.running_var": "bn/moving_variance",
}

BLOCK_MAPS = {
    # order matters!!
    "rep_ncspelan": {
        "rep_ncspelan.conv2.0.": "rep_ncspelan.rep_ncsp_1.",
        "rep_ncspelan.conv2.1.": "rep_ncspelan.conv_block_2.",
        "rep_ncspelan.conv3.0.": "rep_ncspelan.rep_ncsp_2.",
        "rep_ncspelan.conv3.1.": "rep_ncspelan.conv_block_3.",
        ".bottleneck.": ".bottleneck_",
        ".conv1.conv1.": ".rep_conv.conv_block.",
        ".conv1.conv2.": ".rep_conv.pw_conv_block.",
        ".conv1.": ".conv_block_1.",
        ".conv2.": ".conv_block_2.",
        ".conv3.": ".conv_block_3.",
        ".conv4.": ".conv_block_4.",
    },
    "sppelan": {
        ".conv1.": ".conv_block_1.",
        ".conv5.": ".conv_block_2.",
    },
    "multihead_detection": {
        ".heads.": ".detection_head_",
        ".class_conv.0.": ".class_conv.conv_block_1.",
        ".class_conv.1.": ".class_conv.conv_block_2.",
        ".class_conv.2.": ".class_conv.conv.",
        ".anchor_conv.0.": ".anchor_conv.conv_block_1.",
        ".anchor_conv.1.": ".anchor_conv.conv_block_2.",
        ".anchor_conv.2.": ".anchor_conv.conv.",
    },
    "multihead_segmentation": {
        ".detect.heads.": ".detect.detection_head_",
        ".heads.": ".segment.segmentation_head_",
        ".class_conv.0.": ".class_conv.conv_block_1.",
        ".class_conv.1.": ".class_conv.conv_block_2.",
        ".class_conv.2.": ".class_conv.conv.",
        ".anchor_conv.0.": ".anchor_conv.conv_block_1.",
        ".anchor_conv.1.": ".anchor_conv.conv_block_2.",
        ".anchor_conv.2.": ".anchor_conv.conv.",
        ".mask_conv.0.": ".conv_block_1.",
        ".mask_conv.1.": ".conv_block_2.",
        ".mask_conv.2.": ".conv.",
    },
    "adown": {"_adown.conv": "_adown.conv_block_"},
}


def replace_name(weight_name: str, block_map: Dict[str, str]):
    keras_name = weight_name
    for k, v in block_map.items():
        if k in weight_name:
            keras_name = keras_name.replace(k, v)
    return keras_name


def convert_name_torch2keras(torch_name: str, layer_map: Dict[str, str]):
    splitted = torch_name.split(".")
    # torch_ver = ".".join(splitted[0])
    torch_ver = splitted[0]
    keras_ver = layer_map[torch_ver]
    keras_name = re.sub(
        rf"^{torch_ver}.",
        rf"{keras_ver}.",
        torch_name,
    )
    # keras_name = torch_name.replace(torch_ver, keras_ver)
    for k, block_map in BLOCK_MAPS.items():
        if k in keras_ver:
            if k == "multihead_segmentation":
                if re.search(".heads.([0-9]+).(bn|conv).", keras_name):
                    keras_name = re.sub(
                        r".heads.([0-9]+).(bn|conv).",
                        r".segment.segmentation_head_\g<1>.conv_block.\g<2>.",
                        keras_name,
                    )
            keras_name = replace_name(keras_name, block_map)
            if (
                k == "rep_ncspelan"
                and ".conv_block_2." in keras_name
                and "bottleneck_" in keras_name
            ):
                keras_name = keras_name.replace(".conv_block_2.", ".conv_block.")
    torch_var_name = ".".join(keras_name.split(".")[-2:])
    if torch_var_name in VAR_NAME_MAP:
        keras_var_name = VAR_NAME_MAP[torch_var_name]
        keras_name = keras_name.replace(torch_var_name, keras_var_name)
    return keras_name


def get_keras_weights(
    model: Model, torch_state_dict: Dict[str, Tensor], layer_map: Dict[str, str]
) -> Dict[str, ndarray]:
    ported_weights = {}
    count = 0
    keras_weights = [var.path for var in model.variables]
    torch_weights = [
        k
        for k in list(torch_state_dict.keys())
        if "num_batches_tracked" not in k and "anc2vec" not in k
    ]
    for tw in torch_weights:
        kw = convert_name_torch2keras(tw, layer_map)
        if kw in keras_weights:
            layer, variable = kw.split("/")
            k_var = getattr(model.get_layer(layer), variable).numpy()
            if isinstance(k_var, tuple):
                k_shape = k_var[0].shape
            else:
                k_shape = k_var.shape
            ported_weight = torch_state_dict[tw].numpy()
            if len(ported_weight.shape) == 4:
                ported_weight = ported_weight.transpose([2, 3, 1, 0])
            t_shape = ported_weight.shape

            if not k_shape == t_shape:
                raise ValueError(f"Weight Mismatch: {kw}{k_shape} != {tw}{t_shape}")
            ported_weights[kw] = ported_weight
        else:
            count += 1
            continue
    print(f"Numbers of weight not found:{count}")
    return ported_weights


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_weights_path", help="Path to torch weights")
    parser.add_argument("--config_path", help="Path to config file for keras model")
    parser.add_argument("--output_path", help="Path to save keras model")

    args = parser.parse_args()
    return args


def main(args):
    with initialize(version_base=None, config_path=args.config_path):
        config = compose(config_name="config.yaml")

    config = OmegaConf.to_object(config)

    model, layer_map = build_model(config, True, True)
    torch_state_dict = torch.load(args.torch_weights_path,map_location=torch.device('cpu'))

    ported_weights = get_keras_weights(model, torch_state_dict, layer_map)

    for l in model.variables:
        weight = ported_weights.get(l.path, None)
        if weight is None:
            raise ValueError(f"Weight not found for the variable {l.path}")
        l.assign(ops.convert_to_tensor(weight))
    model.save(args.output_path)


if __name__ == "__main__":
    port_args = parse_arguments()
    main(port_args)
