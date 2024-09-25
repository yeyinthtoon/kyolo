import math
from dataclasses import dataclass
from typing import List, Literal, Tuple

from keras import KerasTensor, backend, ops


@dataclass(frozen=True)
class AlignerConf:
    iou: Literal["iou", "diou", "ciou", "siou"]
    iou_factor: float = 6
    cls_factor: float = 0.5
    topk: int = 10


def calculate_iou(
    bbox1: KerasTensor,
    bbox2: KerasTensor,
    metrics: Literal["iou", "diou", "ciou", "siou"] = "iou",
    eps: float = 1e-9,
):
    """
    Calculates IoU (Intersection over Union), DIoU, CIoU, SIoU between bounding boxes.

    Args:
        bbox1: First set of bounding boxes, shape [..., 4] or [..., A, 4].
        bbox2: Second set of bounding boxes, shape [..., 4] or [..., B, 4].
        metrics: The metric to calculate ("iou", "diou", "ciou", "siou"). Defaults to "iou".

    Returns:
        Tensor containing the calculated IoU/DIoU/CIoU.
    """

    bbox1 = ops.cast(bbox1, dtype=backend.floatx())
    bbox2 = ops.cast(bbox2, dtype=backend.floatx())

    # Expand dimensions if necessary for broadcasting
    if len(bbox1.shape) == 2 and len(bbox2.shape) == 2:
        bbox1 = ops.expand_dims(bbox1, axis=1)  # (A, 4) -> (A, 1, 4)
        bbox2 = ops.expand_dims(bbox2, axis=0)  # (B, 4) -> (1, B, 4)
    elif len(bbox1.shape) == 3 and len(bbox2.shape) == 3:
        bbox1 = ops.expand_dims(bbox1, axis=2)  # (B, A, 4) -> (B, A, 1, 4)
        bbox2 = ops.expand_dims(bbox2, axis=1)  # (B, B, 4) -> (B, 1, B, 4)

    # Calculate intersection coordinates
    xmin_inter = ops.maximum(bbox1[..., 0], bbox2[..., 0])
    ymin_inter = ops.maximum(bbox1[..., 1], bbox2[..., 1])
    xmax_inter = ops.minimum(bbox1[..., 2], bbox2[..., 2])
    ymax_inter = ops.minimum(bbox1[..., 3], bbox2[..., 3])

    # Calculate intersection area
    x_min = 0.0
    intersection_area = ops.maximum(xmax_inter - xmin_inter, [x_min]) * ops.maximum(
        ymax_inter - ymin_inter, [x_min]
    )

    # Calculate area of each bbox
    w1 = bbox1[..., 2] - bbox1[..., 0]
    h1 = bbox1[..., 3] - bbox1[..., 1]
    w2 = bbox2[..., 2] - bbox2[..., 0]
    h2 = bbox2[..., 3] - bbox2[..., 1]
    area_bbox1 = w1 * h1
    area_bbox2 = w2 * h2

    # Calculate union area
    union_area = area_bbox1 + area_bbox2 - intersection_area

    # Calculate IoU
    iou = intersection_area / (union_area + eps)

    if metrics == "iou":
        return iou

    # Calculate centroid distance
    cx1 = (bbox1[..., 2] + bbox1[..., 0]) / 2
    cy1 = (bbox1[..., 3] + bbox1[..., 1]) / 2
    cx2 = (bbox2[..., 2] + bbox2[..., 0]) / 2
    cy2 = (bbox2[..., 3] + bbox2[..., 1]) / 2
    cent_dis = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Calculate diagonal length of the smallest enclosing box
    c_x = ops.maximum(bbox1[..., 2], bbox2[..., 2]) - ops.minimum(
        bbox1[..., 0], bbox2[..., 0]
    )
    c_y = ops.maximum(bbox1[..., 3], bbox2[..., 3]) - ops.minimum(
        bbox1[..., 1], bbox2[..., 1]
    )
    if metrics in ["diou", "ciou"]:
        diag_dis = c_x**2 + c_y**2 + eps
        diou = iou - (cent_dis / diag_dis)
        if metrics == "diou":
            return diou

        # Compute aspect ratio penalty term
        arctan = ops.arctan(
            (bbox1[..., 2] - bbox1[..., 0]) / (bbox1[..., 3] - bbox1[..., 1] + eps)
        ) - ops.arctan(
            (bbox2[..., 2] - bbox2[..., 0]) / (bbox2[..., 3] - bbox2[..., 1] + eps)
        )
        v = (4 / (math.pi**2)) * (arctan**2)
        alpha = v / (v - iou + 1 + eps)

        # Compute CIoU
        ciou = diou - alpha * v
        return ciou
    if metrics == "siou":
        sigma = cent_dis**0.5 + eps
        sin_alpha_1 = ops.abs(cx2 - cx1) / sigma
        sin_alpha_2 = ops.abs(cy2 - cy1) / sigma
        threshold = (2**0.5) / 2
        sin_alpha = ops.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = 1 - 2 * ops.sin(ops.arcsin(sin_alpha) - math.pi / 4) ** 2

        rho_x = ((cx2 - cx1) / (c_x + eps)) ** 2
        rho_y = ((cy2 - cy1) / (c_y + eps)) ** 2
        gamma = 2 - angle_cost
        distance_cost = 2 - ops.exp(gamma * rho_x) - ops.exp(gamma * rho_y)

        omiga_w = ops.abs(w1 - w2) / ops.maximum(w1, w2)
        omiga_h = ops.abs(h1 - h2) / ops.maximum(h1, h2)
        shape_cost = ops.power(1 - ops.exp(-1 * omiga_w), 4) + ops.power(
            1 - ops.exp(-1 * omiga_h), 4
        )
        return iou - 0.5 * (distance_cost + shape_cost)
    raise ValueError(f"Metric type {metrics} not supported.")


def get_anchors_and_scalers(
    detection_head_output_shape: List[Tuple[int, int]], input_size: Tuple[int, int]
) -> Tuple[KerasTensor, KerasTensor]:
    def get_box_strides(
        detection_head_output_shape: List[Tuple[int, int]], input_height: int
    ) -> List[int]:
        strides = []
        for h, _ in detection_head_output_shape:
            strides.append(input_height // h)
        return strides

    img_w, img_h = input_size
    strides = get_box_strides(detection_head_output_shape, img_h)
    anchors = []
    scaler = []
    for stride in strides:
        anchor_num = img_w // stride * img_h // stride
        scaler.append(ops.full((anchor_num,), stride))
        shift = stride // 2
        h = ops.arange(0, img_h, stride) + shift
        w = ops.arange(0, img_w, stride) + shift
        anchor_h, anchor_w = ops.meshgrid(h, w, indexing="ij")
        anchor = ops.stack(
            [ops.reshape(anchor_w, -1), ops.reshape(anchor_h, -1)], axis=-1
        )
        anchors.append(anchor)
    anchors = ops.cast(ops.concatenate(anchors, axis=0), dtype=backend.floatx())
    scalers = ops.cast(ops.concatenate(scaler, axis=0), dtype=backend.floatx())
    return anchors, scalers


def get_valid_matrix(anchors: KerasTensor, target_bbox: KerasTensor) -> KerasTensor:
    """
    Calculate target on anchors matrix

    Args:
        anchors: anchors, shape [B, A, 4].
        target_bbox: ground truth boxes, shape [B, T, 4].
    Returns:
        Bool Tensor, shape [B, T, A].
    """
    xmin, ymin, xmax, ymax = ops.split(target_bbox, [1, 2, 3], axis=-1)
    anchors = anchors[None, None]
    anchors_x, anchors_y = ops.split(anchors, [1], axis=-1)

    anchors_x, anchors_y = ops.squeeze(anchors_x, -1), ops.squeeze(anchors_y, -1)

    target_in_x = (xmin < anchors_x) & (anchors_x < xmax)
    target_in_y = (ymin < anchors_y) & (anchors_y < ymax)

    target_on_anchor = target_in_x & target_in_y
    return target_on_anchor


def get_cls_matrix(predict_cls: KerasTensor, target_cls: KerasTensor) -> KerasTensor:
    """
    get target class score of  all anchors

    Args:
        predict_cls: class prediction, shape [B, A, C].
        target_cls: ground truth boxes, shape [B, T, C].
    Returns:
        Tensor, shape [B, T, A].
    """
    predict_cls = ops.sigmoid(predict_cls)
    predict_cls = ops.transpose(predict_cls, (0, 2, 1))
    target_cls = ops.repeat(target_cls, predict_cls.shape[2], 2)
    cls_probabilities = ops.take_along_axis(predict_cls, target_cls, axis=1)
    return cls_probabilities


def filter_topk(
    target_matrix: KerasTensor, topk: int = 10
) -> Tuple[KerasTensor, KerasTensor]:
    """
    filter topk from give matrix and set everything else to zero

    Args:
        target_matrix: shape [B, T, A].
    Returns:
        target_topk_matrix, shape [B, T, A].
        target_topk_mask, shape [B, T, A].
    """
    values, _ = ops.top_k(target_matrix, topk)
    min_v = ops.min(values, axis=-1)[..., None]

    topk_targets = ops.where(
        target_matrix >= min_v, target_matrix, ops.zeros_like(target_matrix)
    )
    topk_masks = topk_targets > 0

    return topk_targets, topk_masks


def filter_duplicates(target_matrix: KerasTensor):
    """
    make sure an anchor only match single target.

    Args:
        target_matrix: shape [B, T, A].
    Returns:
        Tensor, shape [B, A, 1].
    """
    unique_indices = ops.argmax(target_matrix, axis=1)
    return unique_indices[..., None]


def get_align_indices_and_valid_mask(
    predict_cls, predict_bbox, target_cls, target_bbox, anchors, configs: AlignerConf
):
    target_anchor_mask = get_valid_matrix(anchors, target_bbox)

    iou_matrix = ops.clip(
        calculate_iou(target_bbox, predict_bbox, configs.iou), 0.0, 1.0
    )

    cls_matrix = get_cls_matrix(predict_cls, target_cls)

    target_matrix = (
        ops.cast(target_anchor_mask, iou_matrix.dtype)
        * (iou_matrix**configs.iou_factor)
        * (cls_matrix**configs.cls_factor)
    )

    topk_targets, topk_mask = filter_topk(target_matrix, topk=configs.topk)

    aligned_indices = filter_duplicates(topk_targets)

    valid_mask = ops.sum(topk_mask, axis=-2)
    valid_mask = ops.cast(valid_mask, "bool")
    return aligned_indices, valid_mask, target_matrix, iou_matrix


def get_aligned_targets_detection(
    predict_cls,
    predict_bbox,
    target_cls,
    target_bbox,
    number_of_classes,
    anchors,
    configs: AlignerConf,
):
    aligned_indices, valid_mask, target_matrix, iou_matrix = (
        get_align_indices_and_valid_mask(
            predict_cls, predict_bbox, target_cls, target_bbox, anchors, configs
        )
    )

    align_bbox = ops.take_along_axis(
        target_bbox, ops.repeat(aligned_indices, 4, 2), axis=1
    )
    align_cls = ops.squeeze(
        ops.take_along_axis(target_cls, aligned_indices, axis=1), -1
    )
    align_cls = ops.one_hot(align_cls, number_of_classes)

    max_target = ops.amax(target_matrix, axis=-1, keepdims=True)
    max_iou = ops.amax(iou_matrix, axis=-1, keepdims=True)
    normalize_term = (target_matrix / (max_target + 1e-9)) * max_iou
    normalize_term = ops.transpose(normalize_term, (0, 2, 1))
    normalize_term = ops.take_along_axis(normalize_term, aligned_indices, axis=2)

    align_cls = (
        align_cls
        * normalize_term
        * ops.cast(valid_mask, normalize_term.dtype)[:, :, None]
    )

    return align_cls, align_bbox, valid_mask
