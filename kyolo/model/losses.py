import keras
from keras import ops, losses
from kyolo.utils.bounding_box_utils import calculate_iou


def remap_to_batch(valid_mask, values):
    flatten_valid_mask = ops.reshape(valid_mask, [-1])
    true_indices = ops.nonzero(flatten_valid_mask)[0]
    batch_map = ops.scatter_update(
        keras.ops.zeros(ops.shape(flatten_valid_mask), dtype=values.dtype),
        true_indices[..., None],
        values,
    )
    batch_map = ops.reshape(batch_map, ops.shape(valid_mask))
    return batch_map


def bce_yolo(y_true, y_pred):
    bce = losses.binary_crossentropy(y_true, y_pred, from_logits=True, axis=[])
    return ops.sum(bce, axis=[1, 2]) / ops.sum(y_true)


def box_loss_yolo(y_true, y_pred, dtype, iou="ciou"):
    align_box, valid_mask, align_cls = ops.split(y_true, [4, 5], axis=-1)
    valid_mask = ops.cast(valid_mask, bool)
    box_norm = ops.sum(align_cls, axis=-1)[valid_mask[..., 0]]

    valid_mask_box = ops.repeat(valid_mask, 4, -1)
    target_boxes = ops.reshape(align_box[valid_mask_box], (-1, 4))
    predict_boxes = ops.reshape(y_pred[valid_mask_box], (-1, 4))
    iou_value = calculate_iou(predict_boxes, target_boxes, dtype, iou, pairwise=False)

    iou_loss = ((1.0 - iou_value) * box_norm) / ops.sum(align_cls)
    iou_loss = ops.sum(remap_to_batch(valid_mask[..., 0], iou_loss), axis=-1)
    return iou_loss


def dfl_loss_yolo(y_true, y_pred, anchor_norm, reg_max=16):
    align_box, valid_mask, align_cls = ops.split(y_true, [4, 5], axis=-1)
    valid_mask = ops.cast(valid_mask, bool)
    valid_mask_box = ops.repeat(valid_mask, 4, -1)
    box_norm = ops.sum(align_cls, axis=-1)[valid_mask[..., 0]]

    left_target, right_target = ops.split(align_box, 2, axis=-1)
    target_dist = ops.concatenate(
        [(anchor_norm - left_target), (right_target - anchor_norm)], axis=-1
    )
    target_dist = ops.clip(target_dist, 0.0, reg_max - 1.01)

    valid_target_dist = target_dist[valid_mask_box]
    valid_pred_dist = y_pred[valid_mask_box]

    target_left, target_right = (
        ops.floor(valid_target_dist),
        ops.floor(valid_target_dist) + 1,
    )
    weight_left, weight_right = (
        target_right - valid_target_dist,
        valid_target_dist - target_left,
    )
    loss_left = losses.sparse_categorical_crossentropy(
        target_left, valid_pred_dist, from_logits=True
    )
    loss_right = losses.sparse_categorical_crossentropy(
        target_right, valid_pred_dist, from_logits=True
    )
    dfl_loss = loss_left * weight_left + loss_right * weight_right
    dfl_loss = ops.mean(ops.reshape(dfl_loss, (-1, 4)), axis=-1)

    dfl_loss = (dfl_loss * box_norm) / ops.sum(align_cls)
    dfl_loss = ops.sum(remap_to_batch(valid_mask[..., 0], dfl_loss), axis=-1)
    return dfl_loss


def box_loss_yolo_jit(y_true, y_pred, dtype, iou="ciou"):
    align_box, valid_mask, align_cls = ops.split(y_true, [4, 5], axis=-1)
    valid_mask = ops.cast(valid_mask, bool)
    box_norm = ops.sum(align_cls, axis=-1)
    box_norm = ops.where(valid_mask[..., 0], box_norm, 0)
    iou = calculate_iou(align_box, y_pred, dtype, metrics="ciou", pairwise=False)
    iou = ops.where(valid_mask[..., 0], iou, 0)
    iou_loss = ((1.0 - iou) * box_norm) / ops.sum(align_cls)
    iou_loss = ops.sum(iou_loss, axis=-1)
    return iou_loss


def dfl_loss_yolo_jit(y_true, y_pred, anchor_norm, reg_max=16):
    align_box, valid_mask, align_cls = ops.split(y_true, [4, 5], axis=-1)
    valid_mask = ops.cast(valid_mask, bool)

    box_norm = ops.sum(align_cls, axis=-1)
    box_norm = ops.where(valid_mask[..., 0], box_norm, 0)

    left_target, right_target = ops.split(align_box, 2, axis=-1)
    target_dist = ops.concatenate(
        [(anchor_norm - left_target), (right_target - anchor_norm)], axis=-1
    )
    target_dist = ops.clip(target_dist, 0.0, reg_max - 1.01)

    target_left, target_right = ops.floor(target_dist), ops.floor(target_dist) + 1
    weight_left, weight_right = target_right - target_dist, target_dist - target_left

    loss_left = losses.sparse_categorical_crossentropy(
        target_left, y_pred, from_logits=True
    )
    loss_right = losses.sparse_categorical_crossentropy(
        target_right, y_pred, from_logits=True
    )
    dfl_loss = loss_left * weight_left + loss_right * weight_right
    dfl_loss = ops.mean(dfl_loss, axis=-1)
    dfl_loss = (dfl_loss * box_norm) / ops.sum(align_cls)
    dfl_loss = ops.sum(dfl_loss, axis=-1)
    return dfl_loss

