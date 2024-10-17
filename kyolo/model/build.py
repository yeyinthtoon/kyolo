import inspect
from typing import Dict, Union, Tuple, Literal

import keras
from keras import Model
from keras_cv.layers import NonMaxSuppression

from kyolo.model.blocks import BLOCKS_REGISTRY, get_feature_map_shapes
from kyolo.model.layers import ProcessMask, Vec2Box
from kyolo.model.trainer import YoloV9Trainer


def build_model(
    config,
    training: bool = False,
    layer_map_out: bool = False,
) -> Union[Union[Model, YoloV9Trainer], Tuple[Model, Dict]]:
    mask_h, mask_w = 0, 0
    num_classes = config["dataset"]["num_class"]
    model_config = config["model"]["model"]
    common_config = config["common"]
    image_size = common_config["img_size"]
    task = config['task']
    outputs = {}
    if layer_map_out:
        layer_map = {}
        i = 0

    inputs = keras.Input(shape=(image_size, image_size, 3))
    model_layers = {"input": inputs}
    for arch in model_config:
        if arch not in ["backbone", "neck", "head", "prediction", "auxiliary"]:
            continue
        for layer_name, layer_info in model_config[arch].items():
            ln_splitted = layer_name.split("_")
            layer_type = "_".join(ln_splitted[1:])
            block = BLOCKS_REGISTRY[layer_type]
            if layer_name in model_layers.keys():
                raise ValueError(
                    f"Non-unique layer name ({layer_name}) found! Tags must be unique."
                )

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

            is_output = layer_info.get("output", False)
            layer_args = layer_info.get("args", {})
            if (
                "multihead_detection" in layer_type
                or "multihead_segmentation" in layer_type
            ):
                layer_args["num_classes"] = num_classes
                layer_args["reg_max"] = common_config["reg_max"]

            if "name" in set(inspect.signature(block).parameters.keys()):
                layer_args["name"] = f"{arch}.{layer_name}"

            if layer_map_out:
                layer_map[f"{i}"] = f"{arch}.{layer_name}"
                i += 1
            model_layers[layer_name] = block(layer_inputs, **layer_args)

            if is_output:
                tag = ln_splitted[0]
                classes, anchors, boxes = model_layers[layer_name][:3]
                if "multihead_segmentation" in layer_type:
                    shapes = get_feature_map_shapes(layer_inputs[0])
                elif "multihead_detection" in layer_type:
                    shapes = get_feature_map_shapes(layer_inputs)
                else:
                    raise ValueError(f"Unsupported output layer type: {layer_type}")
                boxes = Vec2Box(shapes, (image_size, image_size))(boxes)

                if tag == "main" and training and task == "segmentation":
                    protos = model_layers[layer_name][4]
                    _, mask_h, mask_w, _ = protos.shape
                    
                if tag == "main" and not training:
                    nms_config = config["nms"]
                    nms = NonMaxSuppression(
                        "xyxy",
                        True,
                        confidence_threshold=nms_config["conf_threshold"],
                        iou_threshold=nms_config["iou_threshold"],
                    )(boxes, classes)
                    pred_out = [nms]
                    if "multihead_segmentation" in layer_type:
                        masks = model_layers[layer_name][3]
                        protos = model_layers[layer_name][4]
                        mask_nms = ProcessMask((image_size, image_size))(
                            masks, protos, nms
                        )
                        pred_out.append(mask_nms)
                    outputs[tag] = pred_out
                else:
                    outputs[tag] = [
                        classes,
                        anchors,
                        boxes,
                        *model_layers[layer_name][3:],
                    ]
    if training:
        model = YoloV9Trainer(
            inputs=inputs,
            outputs=outputs,
            head_keys=list(outputs.keys()),
            feature_map_shape=shapes,
            input_size=(image_size, image_size),
            num_of_classes=num_classes,
            iou = config["common"].get("iou", "ciou"),
            reg_max=common_config["reg_max"],
            task=task,
            mask_w=mask_w,
            mask_h=mask_h
        )
    else:
        model = Model(inputs=inputs, outputs=outputs)
    if layer_map_out:
        return model, layer_map

    return model
