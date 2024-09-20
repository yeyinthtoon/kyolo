import math
from typing import List, Tuple

from keras import KerasTensor, ops, backend


def calculate_iou(bbox1, bbox2, metrics="iou"):
    """
    Calculates IoU (Intersection over Union), DIoU, or CIoU between bounding boxes.

    Args:
        bbox1: First set of bounding boxes, shape [..., 4] or [..., A, 4].
        bbox2: Second set of bounding boxes, shape [..., 4] or [..., B, 4].
        metrics: The metric to calculate ("iou", "diou", "ciou"). Defaults to "iou".

    Returns:
        Tensor containing the calculated IoU/DIoU/CIoU.
    """

    metrics = metrics.lower()
    EPS = 1e-9

    bbox1 = ops.cast(bbox1, dtype="float32")
    bbox2 = ops.cast(bbox2, dtype="float32")

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
    area_bbox1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area_bbox2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    # Calculate union area
    union_area = area_bbox1 + area_bbox2 - intersection_area

    # Calculate IoU
    iou = intersection_area / (union_area + EPS)

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
    diag_dis = c_x**2 + c_y**2 + EPS

    diou = iou - (cent_dis / diag_dis)
    if metrics == "diou":
        return diou

    # Compute aspect ratio penalty term
    arctan = ops.arctan(
        (bbox1[..., 2] - bbox1[..., 0]) / (bbox1[..., 3] - bbox1[..., 1] + EPS)
    ) - ops.arctan(
        (bbox2[..., 2] - bbox2[..., 0]) / (bbox2[..., 3] - bbox2[..., 1] + EPS)
    )
    v = (4 / (math.pi**2)) * (arctan**2)
    alpha = v / (v - iou + 1 + EPS)

    # Compute CIoU
    ciou = diou - alpha * v
    return ciou


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


class BoxMatcher:
    def __init__(self, cfg, class_num: int, anchors: KerasTensor) -> None:
        self.class_num = class_num
        self.anchors = anchors
        for attr_name in cfg:
            setattr(self, attr_name, cfg[attr_name])

    def get_valid_matrix(self, target_bbox: KerasTensor):
        xmin, ymin, xmax, ymax = ops.split(target_bbox, [1, 2, 3], axis=-1)
        anchors = self.anchors[None, None]
        anchors_x, anchors_y = ops.split(anchors, [1], axis=-1)

        anchors_x, anchors_y = ops.squeeze(anchors_x, -1), ops.squeeze(anchors_y, -1)

        target_in_x = (xmin < anchors_x) & (anchors_x < xmax)
        target_in_y = (ymin < anchors_y) & (anchors_y < ymax)

        target_on_anchor = target_in_x & target_in_y
        return target_on_anchor

    def get_iou_matrix(self, predict_bbox, target_bbox) -> KerasTensor:
        return ops.clip(calculate_iou(target_bbox, predict_bbox, self.iou), 0, 1)

    def get_cls_matrix(
        self, predict_cls: KerasTensor, target_cls: KerasTensor
    ) -> KerasTensor:
        predict_cls = ops.transpose(predict_cls, (0, 2, 1))
        target_cls = ops.repeat(target_cls, predict_cls.shape[2], 2)
        cls_probabilities = ops.take_along_axis(predict_cls, target_cls, axis=1)
        return cls_probabilities

    def filter_topk(
        self, target_matrix: KerasTensor, topk: int = 10
    ) -> Tuple[KerasTensor, KerasTensor]:
        values, _ = ops.top_k(target_matrix, topk)
        min_v = ops.min(values, axis=-1)[..., None]

        topk_targets = ops.where(
            target_matrix >= min_v, target_matrix, ops.zeros_like(target_matrix)
        )
        topk_masks = topk_targets > 0

        return topk_targets, topk_masks

    def filter_duplicates(self, target_matrix: KerasTensor):
        unique_indices = ops.argmax(target_matrix, axis=1)
        return unique_indices[..., None]

    def __call__(
        self, target: KerasTensor, predict: Tuple[KerasTensor]
    ) -> Tuple[KerasTensor, KerasTensor]:
        predict_cls, predict_bbox = predict
        target_cls, target_bbox = ops.split(target, [1], axis=-1)
        target_cls = ops.maximum(0, ops.cast(target_cls, "int64"))

        grid_mask = self.get_valid_matrix(target_bbox)
        iou_mat = self.get_iou_matrix(predict_bbox, target_bbox)
        cls_mat = self.get_cls_matrix(ops.sigmoid(predict_cls), target_cls)

        grid_mask = ops.cast(grid_mask, iou_mat.dtype)
        target_matrix = (
            grid_mask
            * (iou_mat ** self.factor["iou"])
            * (cls_mat ** self.factor["cls"])
        )
        grid_mask = ops.cast(grid_mask, "bool")

        topk_targets, topk_mask = self.filter_topk(target_matrix, topk=self.topk)

        unique_indices = self.filter_duplicates(topk_targets)

        valid_mask = ops.sum(grid_mask, axis=-2) * ops.sum(topk_mask, axis=-2)
        valid_mask = ops.cast(valid_mask, "bool")

        align_bbox = ops.take_along_axis(
            target_bbox, ops.repeat(unique_indices, 4, 2), axis=1
        )
        align_cls = ops.squeeze(
            ops.take_along_axis(target_cls, unique_indices, axis=1), -1
        )
        align_cls = ops.one_hot(align_cls, self.class_num)

        max_target = ops.amax(target_matrix, axis=-1, keepdims=True)
        max_iou = ops.amax(iou_mat, axis=-1, keepdims=True)
        normalize_term = (target_matrix / (max_target + 1e-9)) * max_iou
        normalize_term = ops.transpose(normalize_term, (0, 2, 1))
        normalize_term = ops.take_along_axis(normalize_term, unique_indices, axis=2)

        valid_mask = ops.cast(valid_mask, "float32")
        align_cls = align_cls * normalize_term * valid_mask[:, :, None]

        return ops.concatenate([align_cls, align_bbox], axis=-1), ops.cast(
            valid_mask, "bool"
        )
