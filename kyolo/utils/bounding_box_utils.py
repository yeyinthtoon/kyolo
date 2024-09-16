from typing import List, Tuple
import math
from keras import KerasTensor, Model, ops


def calculate_iou_tf(bbox1, bbox2, metrics="iou"):
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
    intersection_area = ops.maximum(xmax_inter - xmin_inter, [x_min]) * ops.maximum(ymax_inter - ymin_inter, [x_min])

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
    c_x = ops.maximum(bbox1[..., 2], bbox2[..., 2]) - ops.minimum(bbox1[..., 0], bbox2[..., 0])
    c_y = ops.maximum(bbox1[..., 3], bbox2[..., 3]) - ops.minimum(bbox1[..., 1], bbox2[..., 1])
    diag_dis = c_x**2 + c_y**2 + EPS

    diou = iou - (cent_dis / diag_dis)
    if metrics == "diou":
        return diou

    # Compute aspect ratio penalty term
    arctan = ops.arctan((bbox1[..., 2] - bbox1[..., 0]) / (bbox1[..., 3] - bbox1[..., 1] + EPS)) - ops.arctan(
        (bbox2[..., 2] - bbox2[..., 0]) / (bbox2[..., 3] - bbox2[..., 1] + EPS)
    )
    v = (4 / (math.pi**2)) * (arctan**2)
    alpha = v / (v - iou + 1 + EPS)

    # Compute CIoU
    ciou = diou - alpha * v
    return ciou

def generate_anchors(image_size: List[int], strides: List[int]):
    """
    Find the anchor maps for each w, h.

    Args:
        image_size List: the image size of augmented image size
        strides List[8, 16, 32, ...]: the stride size for each predicted layer

    Returns:
        all_anchors [HW x 2]:
        all_scalers [HW]: The index of the best targets for each anchors
    """
    img_w, img_h = image_size
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
    all_anchors = ops.cast(ops.concatenate(anchors, axis=0), dtype="float32")
    all_scalers = ops.cast(ops.concatenate(scaler, axis=0), dtype="float32")
    return all_anchors, all_scalers


class Vec2Box:
    """
    Decodes bboxes from predictions
    """

    def __init__(
        self,
        model: Model,
        anchor_cfg,
        image_size: List[int],
    ) -> None:

        if hasattr(anchor_cfg, "strides"):
            print(f"🈶 Found stride of model {anchor_cfg.strides}")
            self.strides = anchor_cfg.strides
        else:
            print(
                "🧸 Found no stride of model, ",
                "performed a dummy test for auto-anchor size",
            )
            self.strides = self.create_auto_anchor(model, image_size)

        self.anchor_grid, self.scaler = generate_anchors(image_size, self.strides)

    def create_auto_anchor(self, model: Model, image_size: List[int]):
        dummy_input = ops.zeros(1, 3, *image_size)
        dummy_output = model.predict(dummy_input)
        strides = []
        for predict_head in dummy_output["MAIN"]:
            _, _, *anchor_num = predict_head[2].shape
            strides.append(image_size[1] // anchor_num[1])
        return strides

    def update(self, image_size: List[int]) -> None:
        self.anchor_grid, self.scaler = generate_anchors(image_size, self.strides)

    def __call__(self, preds) -> Tuple[KerasTensor, KerasTensor, KerasTensor]:
        preds_cls, preds_anc, preds_box = [], [], []
        for layer_output in preds:
            pred_cls, pred_anc, pred_box = layer_output
            _, h, w, c = pred_cls.shape
            preds_cls.append(ops.reshape(pred_cls, (-1, h * w, c)))
            _, h, w, r, a = pred_anc.shape
            preds_anc.append(ops.reshape(pred_anc, (-1, h * w, r, a)))
            _, h, w, x = pred_box.shape
            preds_box.append(ops.reshape(pred_box, (-1, h * w, x)))
        preds_cls = ops.concatenate(preds_cls, axis=1)
        preds_anc = ops.concatenate(preds_anc, axis=1)
        preds_box = ops.concatenate(preds_box, axis=1)

        pred_ltrb = preds_box * ops.reshape(self.scaler, (1, -1, 1))
        lt, rb = ops.split(pred_ltrb, 2, axis=-1)
        preds_box = ops.concatenate(
            [self.anchor_grid - lt, self.anchor_grid + rb], axis=-1
        )
        return preds_cls, preds_anc, preds_box
