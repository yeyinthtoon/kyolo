from typing import Callable, Dict, List, Optional, Tuple, TypeAlias, Union

from keras import KerasTensor, activations, layers, ops

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
    # Note: Keras 3 not allow custom padding
    conv_padding = "valid"
    if padding:
        x = layers.ZeroPadding2D(
            padding=auto_pad(kernel_size, kwargs.get("dilation_rate", 1)),
            name=f"{name}.pad" if name else name,
        )(x)

    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=kernel_size,
        padding=conv_padding,
        use_bias=bias,
        name=f"{name}.conv" if name else name,
        **kwargs,
    )(x)
    # Note: set momentum=1-momentum to consistent with pytorch
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
    name: str = "rep_conv",
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
        name=f"{name}.conv_block",
        **kwargs,
    )
    x2 = conv_block(
        inputs,
        out_channels=out_channels,
        kernel_size=1,
        activation=None,
        name=f"{name}.pw_conv_block",
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
    input: KerasTensor,
    out_channels: int,
    kernel_sizes: Tuple[int, int] = (3, 3),
    residual: bool = True,
    expand: float = 1.0,
    name: str = "bottleneck",
    **kwargs,
) -> KerasTensor:
    """Applies a BottleNeck Block with optional residual connection.

    Args:
        input: input tensors
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
    in_channels = input.shape[-1]
    x = rep_conv(
        input,
        out_channels=neck_channels,
        kernel_size=kernel_sizes[0],
        name=f"{name}.rep_conv",
        **kwargs,
    )
    x = conv_block(
        x,
        out_channels=out_channels,
        kernel_size=kernel_sizes[1],
        name=f"{name}.conv_block",
        **kwargs,
    )

    if residual:
        if in_channels != out_channels:
            print(
                f"Residual connection disabled: in_channels ({in_channels}) !=",
                f"out_channels ({out_channels})",
            )
        else:
            x = input + x

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
        name=f"{name}.conv1" if name else name,
        **kwargs,
    )
    x1, x2 = ops.split(x, 2, axis=-1)
    x3 = conv_block(
        x=x2,
        out_channels=process_channels,
        kernel_size=3,
        name=f"{name}.conv2" if name else name,
        **kwargs,
    )
    x4 = conv_block(
        x=x3,
        out_channels=process_channels,
        kernel_size=3,
        name=f"{name}.conv3" if name else name,
        **kwargs,
    )

    x = conv_block(
        x=ops.concatenate([x1, x2, x3, x4], axis=-1),
        out_channels=out_channels,
        kernel_size=1,
        name=f"{name}.conv4" if name else name,
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
        x = layers.ZeroPadding2D(
            padding=auto_pad(kernel_size, kwargs.get("dilation_rate", 1)),
            name=f"{name}.zero_padding" if name else name,
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
def cbfuse(
    x: List[List[KerasTensor]],
    target_tensor: KerasTensor,
    indices: List[int],
    mode: str = "nearest",
) -> KerasTensor:
    # Note: all x and target must have same channel
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
