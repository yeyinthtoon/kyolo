from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias, Union

from keras import KerasTensor, activations, backend, initializers, layers, ops

from kyolo.model.layers import ConstantPadding2D

BLOCKS_REGISTRY: Dict[str, Callable] = {}

Kernel_Size_2D: TypeAlias = Union[int, Tuple[int, int]]


def register_block(block_fn) -> Callable:
    """decorators to register inferencer"""
    if block_fn.__name__ in BLOCKS_REGISTRY:
        raise ValueError(f"Can't register same function twice. {block_fn.__name__}")
    BLOCKS_REGISTRY[block_fn.__name__] = block_fn
    return block_fn


def auto_pad(
    kernel_size: Kernel_Size_2D, dilation: Kernel_Size_2D = 1
) -> Tuple[int, int]:
    """
    Auto Padding for the convolution blocks
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
    return (pad_h, pad_w)


@register_block
def conv_block(
    x: KerasTensor,
    out_channels: int,
    kernel_size: Kernel_Size_2D,
    bias: bool = False,
    eps: float = 1e-3,
    momentum: float = 3e-2,
    activation: Optional[str] = "silu",
    padding: bool = True,
    name: Optional[str] = None,
    **kwargs,
) -> KerasTensor:
    # NOTE: Keras 3 not allow custom padding
    if padding:
        x = layers.ZeroPadding2D(
            padding=auto_pad(kernel_size, kwargs.get("dilation_rate", 1)),
            name=f"{name}.pad" if name else name,
        )(x)

    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=kernel_size,
        padding="valid",
        use_bias=bias,
        name=f"{name}.conv" if name else name,
        **kwargs,
    )(x)
    # NOTE: set momentum=1-momentum to consistent with pytorch
    x = layers.BatchNormalization(
        epsilon=eps, momentum=1 - momentum, name=f"{name}.bn" if name else name
    )(x)
    if activation:
        x = layers.Activation(
            activations.get(activation.lower()), name=f"{name}.act" if name else name
        )(x)
    return x


@register_block
def rep_conv(
    inputs: KerasTensor,
    out_channels: int,
    kernel_size: int = 3,
    activation: Optional[str] = "silu",
    name: Optional[str] = None,
    **kwargs,
) -> KerasTensor:
    """Applies RepConv block
    Applies a Conv Block and followed by Point-wise Conv Block.

    Args:
        inputs: input tensors
        out_channels: integer, the number of output channels expected
        kernel_size: integer, the kernel size of Conv block
        activation: string, the activation function to be used
        name: string, a prefix for names of layers used by the head

    Returns: output tensors from the RepConv block
    """
    x1 = conv_block(
        inputs,
        out_channels=out_channels,
        kernel_size=kernel_size,
        activation=None,
        name=f"{name}.conv_block" if name else name,
        **kwargs,
    )
    x2 = conv_block(
        inputs,
        out_channels=out_channels,
        kernel_size=1,
        activation=None,
        name=f"{name}.pw_conv_block" if name else name,
        **kwargs,
    )
    x = x1 + x2
    if activation:
        x = layers.Activation(
            activations.get(activation.lower()), name=f"{name}.act" if name else name
        )(x)

    return x


@register_block
def bottleneck(
    inputs: KerasTensor,
    out_channels: int,
    kernel_sizes: Tuple[int, int] = (3, 3),
    residual: bool = True,
    expand: float = 1.0,
    name: Optional[str] = None,
    **kwargs,
) -> KerasTensor:
    """Applies a BottleNeck Block with optional residual connection.

    Args:
        inputs: input tensors
        out_channels: integer, the number of output channels expected
        kernel_sizes: tuple of two integers, the first one is the kernel size of
            RepConv block and the last one is the kernel size of Conv block
        residual: bool, whether to use the residual connection
        expand: float, the factor to which the RepConv block channels should be
            expanded
        name: string, a prefix for names of layers used

    Returns: output tensors from the BottleNeck block
    """

    neck_channels = int(out_channels * expand)
    in_channels = inputs.shape[-1]
    x = rep_conv(
        inputs,
        out_channels=neck_channels,
        kernel_size=kernel_sizes[0],
        name=f"{name}.rep_conv" if name else name,
        **kwargs,
    )
    x = conv_block(
        x,
        out_channels=out_channels,
        kernel_size=kernel_sizes[1],
        name=f"{name}.conv_block" if name else name,
        **kwargs,
    )

    if residual:
        if in_channels != out_channels:
            print(
                f"Residual connection disabled: in_channels ({in_channels}) !=",
                f"out_channels ({out_channels})",
            )
        else:
            x = inputs + x

    return x


@register_block
def rep_ncsp(
    inputs: KerasTensor,
    out_channels: int,
    kernel_size: int = 1,
    csp_expand: float = 0.5,
    repeat_num: int = 1,
    neck_args: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    **kwargs,
) -> KerasTensor:
    """Applies RepNCSP block

    Args:
        inputs: input tensors
        out_channels: integer, the number of output channels expected
        kernel_size: integer, the kernel size of Conv block
        csp_expand: float, the factor to which the inputs channels
            should be expanded in the middle layers
        repeat_num: integer, the number of Bottleneck blocks to use
        neck_args: dict, arguments for the Bottleneck blocks
        name: string, a prefix for names of layers used

    Returns: output tensors from the RepNCSP block
    """
    neck_channels = int(out_channels * csp_expand)
    x1 = conv_block(
        inputs,
        out_channels=neck_channels,
        kernel_size=kernel_size,
        name=f"{name}.conv_block_1" if name else name,
        **kwargs,
    )
    for i in range(repeat_num):
        x1 = bottleneck(
            x1,
            neck_channels,
            name=f"{name}.bottleneck_{i}" if name else name,
            **(neck_args if neck_args else {}),
        )

    x2 = conv_block(
        inputs,
        out_channels=neck_channels,
        kernel_size=kernel_size,
        name=f"{name}.conv_block_2" if name else name,
        **kwargs,
    )
    x = ops.concatenate([x1, x2], axis=-1)
    x = conv_block(
        x,
        out_channels=out_channels,
        kernel_size=kernel_size,
        name=f"{name}.conv_block_3" if name else name,
        **kwargs,
    )

    return x


@register_block
def rep_ncspelan(
    inputs: KerasTensor,
    out_channels: int,
    part_channels: int,
    process_channels: Optional[int] = None,
    csp_args: Optional[Dict[str, Any]] = None,
    csp_neck_args: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    **kwargs,
) -> KerasTensor:
    """RepNCSPELAN block combining RepNCSP blocks with ELAN structure.

    Args:
        inputs: input tensors
        out_channels: integer, the number of output channels expected
        part_channels: integer, output channels of the first ConvBlock
        process_channels: integer, channels for RepNCSP blocks
        csp_args: dict, arguments for the RepNCSP block
        csp_neck_args: dict, arguments for the Bottleneck blocks in
            RepNCSP blocks
        name: string, a prefix for names of layers used

    Returns: output tensors from the BottleNeck block
    """
    if process_channels is None:
        process_channels = part_channels // 2
    x = conv_block(
        inputs,
        out_channels=part_channels,
        kernel_size=1,
        name=f"{name}.conv_block_1" if name else name,
        **kwargs,
    )
    x1, x2 = ops.split(x, 2, axis=-1)
    x3 = rep_ncsp(
        x2,
        out_channels=process_channels,
        neck_args=csp_neck_args,
        name=f"{name}.rep_ncsp_1" if name else name,
        **(csp_args if csp_args else {}),
    )
    x3 = conv_block(
        x3,
        out_channels=process_channels,
        kernel_size=3,
        name=f"{name}.conv_block_2" if name else name,
        **kwargs,
    )
    x4 = rep_ncsp(
        x3,
        out_channels=process_channels,
        neck_args=csp_neck_args,
        name=f"{name}.rep_ncsp_2" if name else name,
        **(csp_args if csp_args else {}),
    )
    x4 = conv_block(
        x4,
        out_channels=process_channels,
        kernel_size=3,
        name=f"{name}.conv_block_3" if name else name,
        **kwargs,
    )
    x = ops.concatenate([x1, x2, x3, x4], axis=-1)
    x = conv_block(
        x,
        out_channels=out_channels,
        kernel_size=1,
        name=f"{name}.conv_block_4" if name else name,
        **kwargs,
    )
    return x


@register_block
def elan(
    x: KerasTensor,
    out_channels: int,
    part_channels: int,
    process_channels: Optional[int] = None,
    name: Optional[str] = None,
    **kwargs,
) -> KerasTensor:
    if process_channels is None:
        process_channels = part_channels // 2
    x = conv_block(
        x=x,
        out_channels=part_channels,
        kernel_size=1,
        name=f"{name}.conv_block_1" if name else name,
        **kwargs,
    )
    x1, x2 = ops.split(x, 2, axis=-1)
    x3 = conv_block(
        x=x2,
        out_channels=process_channels,
        kernel_size=3,
        name=f"{name}.conv_block_2" if name else name,
        **kwargs,
    )
    x4 = conv_block(
        x=x3,
        out_channels=process_channels,
        kernel_size=3,
        name=f"{name}.conv_block_3" if name else name,
        **kwargs,
    )

    x = conv_block(
        x=ops.concatenate([x1, x2, x3, x4], axis=-1),
        out_channels=out_channels,
        kernel_size=1,
        name=f"{name}.conv_block_4" if name else name,
        **kwargs,
    )
    return x


@register_block
def pool_block(
    x: KerasTensor,
    method: str = "max",
    kernel_size: int = 2,
    stride: Optional[int] = None,
    padding: bool = True,
    name: Optional[str] = None,
    **kwargs,
) -> KerasTensor:
    method = method.lower()
    pool_classes = {"max": layers.MaxPooling2D, "avg": layers.AveragePooling2D}

    if padding:
        x = ConstantPadding2D(
            padding=auto_pad(kernel_size, kwargs.get("dilation_rate", 1)),
            constant_values=float("-inf") if method == "max" else 0,
            name=f"{name}.padding" if name else name,
        )(x)

    return pool_classes[method](
        pool_size=kernel_size,
        strides=stride,
        padding="valid",
        name=f"{name}.{method}_pooling" if name else name,
        **kwargs,
    )(x)


@register_block
def sppelan(
    x: KerasTensor,
    out_channels: int,
    neck_channels: Optional[int] = None,
    name: Optional[str] = None,
    **kwargs,
) -> KerasTensor:
    neck_channels = neck_channels or out_channels // 2

    features = [
        conv_block(
            x,
            out_channels=neck_channels,
            kernel_size=1,
            bias=False,
            padding=True,
            name=f"{name}.conv_block_1" if name else name,
            **kwargs,
        )
    ]

    for i in range(3):
        features.append(
            pool_block(
                features[-1],
                method="max",
                kernel_size=5,
                stride=1,
                padding=True,
                name=f"{name}.pool_block_{i}" if name else name,
                **kwargs,
            )
        )

    return conv_block(
        ops.concatenate(features, axis=-1),
        out_channels=out_channels,
        kernel_size=1,
        bias=False,
        padding=True,
        name=f"{name}.conv_block_2" if name else name,
        **kwargs,
    )


@register_block
def cb_fuse(
    x: Tuple[List[List[KerasTensor]], KerasTensor],
    indices: List[int],
    mode: str = "nearest",
) -> KerasTensor:
    x, target_tensor = x
    # NOTE: all x and target must have same channel
    allow_modes = set(["bilinear", "nearest", "bicubic", "lanczos3", "lanczos5"])
    assert len(x) == len(indices)
    if mode.lower() not in allow_modes:
        raise ValueError(f"only {allow_modes} are allow for mode")
    _, h, w, _ = target_tensor.shape
    outs = []
    for i, tensorlist in zip(indices, x):
        outs.append(
            ops.image.resize(
                tensorlist[i],
                (h, w),
                interpolation=mode.lower(),
            )
        )
    outs.append(target_tensor)

    return ops.sum(ops.stack(outs), axis=0)


def anc2vec(x: KerasTensor, regmax: int = 16) -> Tuple[KerasTensor, KerasTensor]:
    kernel = ops.reshape(
        ops.arange(start=0, stop=regmax, dtype=backend.floatx()), (1, 1, 1, regmax, 1)
    )

    _, h, w, c = x.shape
    x = ops.reshape(x, (-1, h, w, 4, c // 4))
    vector_x = ops.softmax(x, -1)
    vector_x = ops.conv(vector_x, kernel)[:, :, :, :, 0]

    return x, vector_x


def round_up(x: int, div: int = 1) -> int:
    return x + (-x % div)


def conv_sequence(
    x: KerasTensor,
    inter_channels: int,
    out_channels: int,
    groups: int,
    bias_init: float,
    name: Optional[str] = None,
):
    x = conv_block(x, inter_channels, 3, name=f"{name}.conv_block_1" if name else name)
    x = conv_block(
        x,
        inter_channels,
        3,
        groups=groups,
        name=f"{name}.conv_block_2" if name else name,
    )

    x = layers.Conv2D(
        out_channels,
        1,
        groups=groups,
        bias_initializer=initializers.Constant(bias_init),
        name=f"{name}.conv" if name else name,
    )(x)
    return x


def detection(
    x: KerasTensor,
    num_classes: int,
    anchor_neck_channels: int,
    class_neck_channles: int,
    reg_max: int = 16,
    use_group: bool = True,
    name: Optional[str] = None,
):
    groups = 4 if use_group else 1
    anchor_channels = 4 * reg_max

    anchor_x = conv_sequence(
        x,
        anchor_neck_channels,
        anchor_channels,
        groups,
        1.0,
        name=f"{name}.anchor_conv" if name else name,
    )
    # NOTE: different from ultralytic bias initialization
    class_x = conv_sequence(
        x,
        class_neck_channles,
        num_classes,
        1,
        -10.0,
        name=f"{name}.class_conv" if name else name,
    )
    anchor_x, vector_x = anc2vec(anchor_x, regmax=reg_max)
    return class_x, anchor_x, vector_x


@register_block
def mulithead_detection(
    x: List[KerasTensor],
    num_classes: int,
    reg_max: Union[int, List[int]] = 16,
    use_group: bool = True,
    name: Optional[str] = None,
    **kwargs,
):
    if not isinstance(reg_max, list):
        reg_max = [reg_max] * len(x)
    min_c = x[0].shape[-1]
    groups = 4 if use_group else 1

    outs = []
    for i, (x_in, r) in enumerate(zip(x, reg_max)):
        anchor_channels = 4 * r
        anchor_neck = max(round_up(min_c // 4, groups), anchor_channels, 16)
        class_neck = max(min_c, min(num_classes * 2, 128))
        outs.append(
            detection(
                x_in,
                num_classes,
                anchor_neck,
                class_neck,
                r,
                use_group,
                name=f"{name}.detection_head_{i}" if name else name,
                **kwargs,
            )
        )
    return outs


@register_block
def multihead_segmentation(
    inputs: Tuple[List[List[KerasTensor]], KerasTensor],
    num_classes: int,
    num_masks: int,
    reg_max: Union[int, List[int]] = 16,
    name: Optional[str] = None,
    use_group: bool = True,
    **det_kwargs,
) -> KerasTensor:
    """Mutlihead Segmentation module for Dual segment or Triple segment

    Args:
        inputs: list of keras tensors, list of input tensors
        num_classes: integer, number of classes for the model to predict
        num_masks: integer, number of output masks for the model
        name: string, a prefix for names of layers used

    Returns: output tensors from the BottleNeck block
    """
    proto_in = inputs[1]
    min_c = 999
    for x_in in inputs[0]:
        min_c = min(min_c, x_in.shape[-1])

    outs = [
        mulithead_detection(
            inputs[0],
            num_classes=num_classes,
            reg_max=reg_max,
            use_group= use_group,
            name= f"{name}.detect" if name else name,
            **det_kwargs,
        )
    ]
    for i, x_in in enumerate(inputs[0]):
        mask_neck = max(min_c // 4, num_masks)
        outs.append(
            conv_sequence(
                x_in,
                inter_channels=mask_neck,
                out_channels=num_masks,
                groups=1,
                bias_init=1.0,
                name=f"{name}.segment.segmentation_head_{i}" if name else name,
            )
        )
    outs.append(
        conv_block(
            proto_in,
            out_channels=num_masks,
            kernel_size=1,
            name=f"{name}.segment.segmentation_head_{len(outs)-1}.conv_block" if name else name,
        )
    )
    return outs



@register_block
def aconv(x: KerasTensor, out_channels: int, name: Optional[str] = None):
    """Downsampling module combining average pooling with convolution."""
    x = pool_block(
        x, "avg", kernel_size=2, stride=1, name=f"{name}.pool" if name else name
    )
    x = conv_block(
        x,
        out_channels=out_channels,
        kernel_size=3,
        strides=2,
        name=f"{name}.conv_block" if name else name,
    )

    return x


@register_block
def adown(x: KerasTensor, out_channels: int, name: Optional[str] = None):
    half_out_channels = out_channels // 2
    x = pool_block(
        x, "avg", kernel_size=2, stride=1, name=f"{name}.pool_block_1" if name else name
    )
    x1, x2 = ops.split(x, 2, axis=3)
    x1 = conv_block(
        x1,
        out_channels=half_out_channels,
        kernel_size=3,
        strides=2,
        name=f"{name}.conv_block_1" if name else name,
    )
    x2 = pool_block(
        x2,
        "max",
        kernel_size=3,
        stride=2,
        name=f"{name}.pool_block_2" if name else name,
    )
    x2 = conv_block(
        x2,
        out_channels=half_out_channels,
        kernel_size=1,
        name=f"{name}.conv_block_2" if name else name,
    )
    return ops.concatenate((x1, x2), axis=3)


@register_block
def cb_linear(
    x: KerasTensor,
    out_channels: List[int],
    kernel_size: int = 1,
    name: Optional[str] = None,
    padding: bool = True,
    **kwargs,
):
    total_out_channels = sum(out_channels)
    channel_indices = [0] * len(out_channels)

    for index, out_channel in enumerate(out_channels):
        channel_indices[index] = channel_indices[index - 1] + out_channel

    if padding:
        x = layers.ZeroPadding2D(
            padding=auto_pad(kernel_size, kwargs.get("dilation_rate", 1)),
            name=f"{name}.pad" if name else name,
        )(x)
    x = layers.Conv2D(
        filters=total_out_channels,
        kernel_size=kernel_size,
        padding="valid",
        name=f"{name}.conv" if name else name,
        **kwargs,
    )(x)
    x = ops.split(x, channel_indices, axis=3)

    return x


@register_block
def concat(xs: List[KerasTensor], axis: int = -1) -> KerasTensor:
    return ops.concatenate(xs, axis=axis)


@register_block
def upsample(
    x: KerasTensor,
    scale_factor: Kernel_Size_2D,
    mode: str = "nearest",
    name: Optional[str] = None,
    **kwargs,
):
    return layers.UpSampling2D(
        size=scale_factor, interpolation=mode, name=name, **kwargs
    )(x)


def flatten_predictions(
    inputs: List[Tuple[KerasTensor, KerasTensor, KerasTensor]],
) -> Tuple[KerasTensor, KerasTensor, KerasTensor]:
    cls = []
    anchors = []
    boxes = []
    for pred_cls, pred_anc, pred_box in inputs:
        _, h, w, c = pred_cls.shape
        cls.append(ops.reshape(pred_cls, (-1, h * w, c)))
        _, h, w, r, a = pred_anc.shape
        anchors.append(ops.reshape(pred_anc, (-1, h * w, r, a)))
        _, h, w, x = pred_box.shape
        boxes.append(ops.reshape(pred_box, (-1, h * w, x)))
    cls = ops.concatenate(cls, axis=1)
    anchors = ops.concatenate(anchors, axis=1)
    boxes = ops.concatenate(boxes, axis=1)
    return cls, anchors, boxes


def get_feature_map_shapes(
    inputs: List[Tuple[KerasTensor, KerasTensor, KerasTensor]],
) -> List[Tuple[int, int]]:
    shapes = []
    for _, _, pred_box in inputs:
        _, h, w, _ = pred_box.shape
        shapes.append((h, w))
    return shapes
