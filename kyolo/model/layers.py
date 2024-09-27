from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

from keras import KerasTensor
from keras.src import backend, ops
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.utils import argument_validation
from kyolo.utils.bounding_box_utils import get_anchors_and_scalers
from kyolo.utils.mask_utils import get_mask

Kernel_Size_2D: TypeAlias = Union[int, Tuple[int, int]]


class ConstantPadding2D(Layer):
    def __init__(
        self,
        padding: Kernel_Size_2D = (1, 1),
        constant_values: Optional[Union[int, float]] = None,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_format = backend.standardize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, "__len__"):
            if len(padding) != 2:
                raise ValueError(
                    "`padding` should have two elements. "
                    f"Received: padding={padding}."
                )
            height_padding = argument_validation.standardize_tuple(
                padding[0], 2, "1st entry of padding", allow_zero=True
            )
            width_padding = argument_validation.standardize_tuple(
                padding[1], 2, "2nd entry of padding", allow_zero=True
            )
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError(
                "`padding` should be either an int, a tuple of 2 ints "
                "(symmetric_height_crop, symmetric_width_crop), "
                "or a tuple of 2 tuples of 2 ints "
                "((top_crop, bottom_crop), (left_crop, right_crop)). "
                f"Received: padding={padding}."
            )
        self.input_spec = InputSpec(ndim=4)
        self.constant_values = constant_values

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        output_shape = list(input_shape)
        spatial_dims_offset = 2 if self.data_format == "channels_first" else 1
        for index in range(0, 2):
            if output_shape[index + spatial_dims_offset] is not None:
                output_shape[index + spatial_dims_offset] += (
                    self.padding[index][0] + self.padding[index][1]
                )
        return tuple(output_shape)

    def call(self, inputs: KerasTensor) -> KerasTensor:
        if self.data_format == "channels_first":
            all_dims_padding = ((0, 0), (0, 0), *self.padding)
        else:
            all_dims_padding = ((0, 0), *self.padding, (0, 0))
        return ops.pad(inputs, all_dims_padding, constant_values=self.constant_values)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "padding": self.padding,
            "data_format": self.data_format,
            "constant_values": self.constant_values,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class Vec2Box(Layer):
    def __init__(
        self,
        detection_head_output_shape: List[Tuple[int, int]],
        input_size: Tuple[int, int],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.detection_head_output_shape = detection_head_output_shape
        self.input_size = input_size
        self.anchors, self.scalers = get_anchors_and_scalers(
            detection_head_output_shape, input_size
        )

    def call(self, inputs: KerasTensor) -> KerasTensor:
        pred_ltrb = inputs * ops.reshape(self.scalers, (1, -1, 1))
        lt, rb = ops.split(pred_ltrb, 2, axis=-1)
        preds_box = ops.concatenate([self.anchors - lt, self.anchors + rb], axis=-1)
        return preds_box

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = {
            "detection_head_output_shape": self.detection_head_output_shape,
            "input_size": self.input_size,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class ProcessMask(Layer):
    """
    Layer for processing the segmentation masks from post NMS bounding boxes and model output
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        max_detection: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.max_detection = max_detection

    def call(
        self, masks: KerasTensor, protos: KerasTensor, nms: Dict[str, KerasTensor]
    ) -> KerasTensor:

        pred_bbox = nms["boxes"]
        pred_masks = ops.take_along_axis(
            masks, ops.expand_dims(nms["idx"], axis=-1), axis=1
        )
        masks_nms = get_mask(pred_masks, protos, pred_bbox, self.img_size)
        masks_nms = ops.where(masks_nms > 0.5, masks_nms, 0)

        return masks_nms

    def compute_output_shape(self, input_shape: Tuple) -> Tuple[int, ...]:
        output_shape = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            self.max_detection,
        )
        return output_shape

    def get_config(self) -> Dict[str, Any]:
        config = {
            "max_detection": self.max_detection,
            "img_size": self.img_size,
        }
        base_config = super().get_config()
        return {**base_config, **config}
