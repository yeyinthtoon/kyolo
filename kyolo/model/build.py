import inspect
from typing import Dict, Optional, Union

import keras
from keras import Model
from keras_cv.layers import NonMaxSuppression
from kyolo.model.blocks import (BLOCKS_REGISTRY, flatten_predictions,
                                get_feature_map_shapes)
from kyolo.model.layers import ProcessMask, Vec2Box


def build_model(
    config, training: bool = False, layer_map_out: bool = False
) -> Union[Model, Optional[Dict]]:
    num_classes = config["num_class"]
    model_config = config["model"]
    anchor_config = config["anchor"]
    image_size = config["img_size"]
    outputs = {}
    if layer_map_out:
        layer_map = {}
        i = 0
    tags = []

    inputs = keras.Input(shape=(image_size, image_size, 3))
    model_layers = {"input": inputs}
    for arch in model_config.keys():
        for layer_spec in model_config[arch]:
            layer_type, layer_info = next(iter(layer_spec.items()))
            if len(layer_info["inputs"]) > 1:
                layer_inputs = []
                for l_input in layer_info["inputs"]:
                    if isinstance(l_input, list):
                        layer_inputs.append(
                            [model_layers[sub_input] for sub_input in l_input]
                        )
                    else:
                        layer_inputs.append(model_layers[l_input])
            else:
                layer_inputs = model_layers[layer_info["inputs"][0]]
            tag = layer_info["tag"]
            if tag in tags:
                raise ValueError(f"Non-unique tag ({tag}) found! Tags must be unique.")

            tags.append(tag)
            is_output = layer_info.get("output", False)

            layer_args = layer_info.get("args", {})
            if (
                "multihead_detection" in layer_type
                or "multihead_segmentation" in layer_type
            ):
                layer_args["num_classes"] = num_classes
                layer_args["reg_max"] = anchor_config["reg_max"]

            block = BLOCKS_REGISTRY[layer_type]

            if "name" in set(inspect.signature(block).parameters.keys()):
                layer_args["name"] = f"{arch}.{tag}_{layer_type}"

            if layer_map_out:
                layer_map[f"{i}"] = f"{arch}.{tag}_{layer_type}"
                i += 1
            model_layers[tag] = block(layer_inputs, **layer_args)

            if is_output:
                if tag == "main" and not training:
                    if "multihead_segmentation" in layer_type:
                        box_out = model_layers[tag][0]
                        mask_out = model_layers[tag][1:]
                        shapes = get_feature_map_shapes(box_out)
                        classes, _, boxes = flatten_predictions(box_out)
                        boxes = Vec2Box(shapes, (image_size, image_size))(boxes)
                        nms = NonMaxSuppression(
                            "xyxy", True, confidence_threshold=config["min_nms_conf"]
                        )(boxes, classes)
                        mask_nms = ProcessMask((image_size, image_size))(mask_out, nms)
                        pred_out = [nms, mask_nms]
                    elif "multihead_detection" in layer_type:
                        box_out = model_layers[tag]
                        shapes = get_feature_map_shapes(box_out)
                        classes, _, boxes = flatten_predictions(box_out)
                        boxes = Vec2Box(shapes, (image_size, image_size))(boxes)
                        nms = NonMaxSuppression(
                            "xyxy", True, confidence_threshold=config["min_nms_conf"]
                        )(boxes, classes)
                        pred_out = nms
                    else:
                        raise ValueError(
                            "Only segmentation and detection heads output are allowed"
                        )
                    outputs[tag] = pred_out
                else:
                    outputs[tag] = model_layers[tag]
    model = Model(inputs=inputs, outputs=outputs)
    if layer_map_out:
        return model, layer_map

    return model
