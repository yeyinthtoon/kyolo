from typing import Tuple

from keras import KerasTensor
from keras.src import backend, ops


def get_mask(
    masks: KerasTensor,
    protos: KerasTensor,
    bboxes: KerasTensor,
    img_size: Tuple[int, int],
) -> KerasTensor:
    """
    Calculate the mask from the mask and proto outputs of the model and
    put every value in the mask outside the corresponding boxes to zero.

    Args:
        masks: KerasTensor, mask output from model in the shape [B,num_masks, num_boxes]
        protos: KerasTensor, proto output from model in the shape [B,H,W,num_boxes]
        bboxes: KerasTensor, bounding boxes in the shape of [B,W,H,num_boxes]
        img_size: list of int, input image size of the model

    Returns: processed and filtered masks in the shape [B,W,H,num_boxes]
    """
    _, mask_w, mask_h, channel = protos.shape
    batch = ops.shape(protos)[0]
    protos = ops.reshape(protos, (batch, -1, channel))
    masks = ops.transpose(masks, (0, 2, 1))
    masks = ops.reshape(
        ops.sigmoid(ops.matmul(protos, masks)), (batch, mask_w, mask_h, -1)
    )

    # rescale bboxes to match the mask size
    scale = ops.tile(ops.array([mask_w / img_size[0], mask_h / img_size[1]]), (1, 2))
    bboxes = bboxes * scale

    y, x = ops.meshgrid(
        ops.arange(mask_h, dtype=backend.floatx()),
        ops.arange(mask_w, dtype=backend.floatx()),
        indexing="ij",
    )
    detected_num = ops.shape(masks)[-1]

    x = x[None, :, :, None]
    y = y[None, :, :, None]
    x = ops.tile(x, [batch, 1, 1, detected_num])
    y = ops.tile(y, [batch, 1, 1, detected_num])

    xmin, ymin, xmax, ymax = ops.unstack(bboxes, axis=-1)

    # Reshape for broadcasting
    xmin = ops.reshape(xmin, [batch, 1, 1, detected_num])
    ymin = ops.reshape(ymin, [batch, 1, 1, detected_num])
    xmax = ops.reshape(xmax, [batch, 1, 1, detected_num])
    ymax = ops.reshape(ymax, [batch, 1, 1, detected_num])

    mask_filter = ops.logical_and(
        ops.logical_and(x >= xmin, x < xmax), ops.logical_and(y >= ymin, y < ymax)
    )
    mask_filter = ops.cast(mask_filter, masks.dtype)
    filtered_masks = masks * mask_filter

    return filtered_masks


def generate_bbox_mask(bboxes, mask_h, mask_w, image_h, image_w, dtype):
    scale = ops.array(
        [mask_w / image_w, mask_h / image_h, mask_w / image_w, mask_h / image_h]
    )

    bboxes = bboxes * scale
    y, x = ops.meshgrid(
        ops.arange(mask_h, dtype=dtype),
        ops.arange(mask_w, dtype=dtype),
        indexing="ij",
    )
    max_detect = ops.shape(bboxes)[1]
    batch = ops.shape(bboxes)[0]
    x = x[None, :, :, None]
    y = y[None, :, :, None]

    x = ops.broadcast_to(x, [batch, mask_h, mask_w, max_detect])
    y = ops.broadcast_to(y, [batch, mask_h, mask_w, max_detect])

    xmin, ymin, xmax, ymax = ops.unstack(bboxes, axis=-1)

    # Reshape for broadcasting
    xmin = xmin[:, None, None, :]
    ymin = ymin[:, None, None, :]
    xmax = xmax[:, None, None, :]
    ymax = ymax[:, None, None, :]
    mask = ops.logical_and(
        ops.logical_and(x >= xmin, x < xmax), ops.logical_and(y >= ymin, y < ymax)
    )
    mask = ops.cast(mask, dtype)
    return mask
