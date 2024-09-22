import argparse
import json

import torch
from keras import ops
from kyolo.model.build import build_model
from kyolo.utils.porting_utils import get_keras_weights


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_weights_path", help="Path to torch weights")
    parser.add_argument("--config_path", help="Path to config file for keras model")
    parser.add_argument("--output_path", help="Path to save keras model")

    args = parser.parse_args()
    return args


def main(args):
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model, layer_map = build_model(config, True, True)
    torch_state_dict = torch.load(args.torch_weights_path)

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
