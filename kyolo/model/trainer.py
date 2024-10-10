from dataclasses import asdict
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple

from keras import Model, ops

from kyolo.utils.bounding_box_utils import (
    AlignerConf,
    get_aligned_targets_detection,
    get_anchors_and_scalers,
)


class YoloV9Trainer(Model):
    def __init__(
        self,
        head_keys: List[str],
        feature_map_shape: List[Tuple[int, int]],
        input_size: Tuple[int, int],
        num_of_classes: int,
        aligner_config: AlignerConf,
        reg_max: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.head_keys = head_keys
        anchors, scalers = get_anchors_and_scalers(feature_map_shape, input_size)
        self.anchors = ops.cast(anchors, self.dtype)
        self.scalers = ops.cast(scalers, self.dtype)
        self.anchor_norm = self.anchors / self.scalers[..., None]
        self.num_of_classes = num_of_classes
        self.get_aligned_targets_detection = partial(
            get_aligned_targets_detection, **asdict(aligner_config)
        )

        self.reg_max = reg_max

    def compile(
        self,
        box_loss,
        classification_loss,
        dfl_loss,
        box_loss_weight: float,
        classification_loss_weight: float,
        dfl_loss_weight: float,
        box_loss_iou: Literal["iou", "diou", "ciou", "siou"] = "ciou",
        head_loss_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        head_loss_weights = {} if not head_loss_weights else head_loss_weights
        losses = {}
        loss_weights = {}
        for head_key in self.head_keys:
            head_loss_weight = head_loss_weights.get(head_key, 1.0)
            losses[f"{head_key}_box"] = partial(
                box_loss, dtype=self.dtype, iou=box_loss_iou
            )
            losses[f"{head_key}_class"] = classification_loss
            losses[f"{head_key}_dfl"] = partial(
                dfl_loss, anchor_norm=self.anchor_norm, reg_max=self.reg_max
            )
            loss_weights[f"{head_key}_box"] = box_loss_weight * head_loss_weight
            loss_weights[f"{head_key}_class"] = (
                classification_loss_weight * head_loss_weight
            )
            loss_weights[f"{head_key}_dfl"] = dfl_loss_weight * head_loss_weight

        self.yolo_loss_weights = loss_weights
        super().compile(loss=losses, **kwargs)

    def compute_loss(self, x, y, y_pred, sample_weight=None, **kwargs):
        del sample_weight
        y_pred_final = {}
        y_true_final = {}
        for head_key in self.head_keys:
            cls, anchors, boxes = (
                ops.cast(y_pred[head_key][0], self.dtype),
                ops.cast(y_pred[head_key][1], self.dtype),
                ops.cast(y_pred[head_key][2], self.dtype),
            )
            align_cls, align_bbox, valid_mask, _ = self.get_aligned_targets_detection(
                cls,
                boxes,
                y["classes"],
                y["bboxes"],
                self.num_of_classes,
                self.anchors,
                self.dtype,
            )
            align_bbox = align_bbox / self.scalers[None, ..., None]
            boxes = boxes / self.scalers[None, ..., None]

            y_pred_final[f"{head_key}_box"] = boxes
            y_pred_final[f"{head_key}_dfl"] = anchors
            y_pred_final[f"{head_key}_class"] = cls

            box_target = ops.concatenate(
                [align_bbox, valid_mask[..., None], align_cls], axis=-1
            )

            y_true_final[f"{head_key}_box"] = box_target
            y_true_final[f"{head_key}_dfl"] = box_target
            y_true_final[f"{head_key}_class"] = align_cls
        return super().compute_loss(
            x=x,
            y=y_true_final,
            y_pred=y_pred_final,
            sample_weight=self.yolo_loss_weights,
            **kwargs,
        )
