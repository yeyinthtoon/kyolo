from dataclasses import asdict
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple

from keras import Model, ops

from kyolo.utils.bounding_box_utils import (
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
        reg_max: int,
        iou: Literal["iou", "diou", "ciou", "siou"],
        iou_factor: float = 6,
        cls_factor: float = 0.5,
        topk: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.head_keys = head_keys
        self.feature_map_shape = feature_map_shape
        self.input_size = input_size
        self.num_of_classes = num_of_classes
        # self.aligner_config = aligner_config
        self.reg_max = reg_max
        self.iou = iou
        self.iou_factor = iou_factor
        self.cls_factor = cls_factor
        self.topk = topk
        anchors, scalers = get_anchors_and_scalers(feature_map_shape, input_size)
        self.anchors = ops.cast(anchors, self.dtype)
        self.scalers = ops.cast(scalers, self.dtype)
        self.anchor_norm = self.anchors / self.scalers[..., None]
        self.get_aligned_targets_detection = partial(
            get_aligned_targets_detection,
            iou = iou,
            iou_factor = iou_factor,
            cls_factor = cls_factor,
            topk = topk,
        )

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
        dtype = self.dtype
        anchor_norm = self.anchor_norm
        reg_max = self.reg_max
        def _box_loss(x,y):
            return box_loss(x,y,dtype=dtype, iou=box_loss_iou)
        def _dfl_loss(*x):
            return dfl_loss(*x, anchor_norm=anchor_norm, reg_max=reg_max)
        for head_key in self.head_keys:
            head_loss_weight = head_loss_weights.get(head_key, 1.0)
            losses[f"{head_key}_box"] = _box_loss
            losses[f"{head_key}_class"] = classification_loss
            losses[f"{head_key}_dfl"] = _dfl_loss
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "head_keys" : self.head_keys,
            "feature_map_shape" : self.feature_map_shape,
            "input_size" : self.input_size,
            "num_of_classes" : self.num_of_classes,
            "iou" : self.iou,
            "iou_factor" : self.iou_factor,
            "cls_factor" : self.cls_factor,
            "topk" : self.topk,
            "reg_max" : self.reg_max,
        })
        return config
