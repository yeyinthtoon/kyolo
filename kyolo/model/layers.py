from typing import Any, Dict, Optional, Tuple, TypeAlias, Union

from keras import KerasTensor
from keras.src import backend, ops
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.utils import argument_validation

Kernel_Size_2D: TypeAlias = Union[int, Tuple[int, int]]


class ConstantPadding2D(Layer):
    def __init__(
        self,
        padding: Kernel_Size_2D = (1, 1),
        constant_values: Optional[int] = None,
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
