from dataclasses import asdict
from functools import partial
from typing import Callable, Dict, List, Literal, Optional, Tuple

from keras import Model, ops, saving, activations

from kyolo.utils.bounding_box_utils import (
    get_aligned_targets_detection,
    get_anchors_and_scalers,
    get_normalized_box_area,
    generate_bbox_mask,
)


class YoloV9Trainer(Model):
    def __init__(
        self,
        model,
        head_keys: List[str],
        feature_map_shape: List[Tuple[int, int]],
        input_size: Tuple[int, int],
        num_of_classes: int,
        reg_max: int,
        task: Literal["detection", "segmentation"],
        mask_h: int,
        mask_w: int,
        iou: Literal["iou", "diou", "ciou", "siou"],
        iou_factor: float = 6,
        cls_factor: float = 0.5,
        topk: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.head_keys = head_keys
        self.feature_map_shape = feature_map_shape
        self.input_size = input_size
        self.num_of_classes = num_of_classes
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
            iou=iou,
            iou_factor=iou_factor,
            cls_factor=cls_factor,
            topk=topk,
        )
        self.task = task
        if task == "segmentation" and (mask_h <= 0 or mask_w <= 0):
            raise ValueError("mask shape not valid")
        self.mask_h = mask_h
        self.mask_w = mask_w
        self.reg_max = reg_max

    def call(self, *args, **kwargs):
        return self.model.call(*args, **kwargs)

    def compile(
        self,
        box_loss,
        classification_loss,
        dfl_loss,
        box_loss_weight: float,
        classification_loss_weight: float,
        dfl_loss_weight: float,
        box_loss_iou: Literal["iou", "diou", "ciou", "siou"] = "ciou",
        segmentation_loss: Optional[Callable] = None,
        segmentation_loss_weight: Optional[float] = None,
        head_loss_weights: Optional[Dict[str, float]] = None,
        loss_reduction: str = "sum",
        **kwargs,
    ):
        head_loss_weights = {} if not head_loss_weights else head_loss_weights
        losses = {}
        loss_weights = {}

        for head_key in self.head_keys:
            head_loss_weight = head_loss_weights.get(head_key, 1.0)
            losses[f"{head_key}_box"] = box_loss(
                iou=box_loss_iou,
                reduction=loss_reduction,
            )
            losses[f"{head_key}_class"] = classification_loss(
                from_logits=False, reduction=loss_reduction
            )
            losses[f"{head_key}_dfl"] = dfl_loss(
                self.anchor_norm,
                self.reg_max,
                reduction=loss_reduction,
            )
            loss_weights[f"{head_key}_box"] = box_loss_weight * head_loss_weight
            loss_weights[f"{head_key}_class"] = (
                classification_loss_weight * head_loss_weight
            )
            loss_weights[f"{head_key}_dfl"] = dfl_loss_weight * head_loss_weight
            if self.task == "segmentation":
                if not segmentation_loss:
                    raise ValueError("missing segmentation loss")
                losses[f"{head_key}_segmentation"] = segmentation_loss(
                    reduction=loss_reduction
                )
                segmentation_loss_weight = (
                    box_loss_weight
                    if not segmentation_loss_weight
                    else segmentation_loss_weight
                )
                loss_weights[f"{head_key}_segmentation"] = (
                    segmentation_loss_weight * head_loss_weight
                )

        self.yolo_loss_weights = loss_weights
        super().compile(loss=losses, **kwargs)

    def compute_loss(self, x, y, y_pred, sample_weight=None, **kwargs):
        y_pred_final = {}
        y_true_final = {}
        sample_weights = {}
        for head_key in self.head_keys:
            cls, anchors, boxes = (
                ops.cast(y_pred[head_key][0], self.dtype),
                ops.cast(y_pred[head_key][1], self.dtype),
                ops.cast(y_pred[head_key][2], self.dtype),
            )

            align_cls, align_bbox, valid_mask, aligned_indices = (
                self.get_aligned_targets_detection(
                    ops.stop_gradient(cls),
                    ops.stop_gradient(boxes * self.scalers[..., None]),
                    y["classes"],
                    y["bboxes"],
                    self.num_of_classes,
                    self.anchors,
                    self.dtype,
                    from_logits=False,
                )
            )

            align_bbox_scaled = align_bbox / self.scalers[..., None]
            valid_align_bbox = align_bbox_scaled * valid_mask[..., None]

            boxes = boxes * valid_mask[..., None]

            cls_norm = ops.maximum(ops.sum(align_cls), 1.0)
            box_norm = ops.sum(align_cls, axis=-1) * valid_mask

            y_pred_final[f"{head_key}_box"] = boxes
            y_pred_final[f"{head_key}_dfl"] = anchors
            y_pred_final[f"{head_key}_class"] = cls

            y_true_final[f"{head_key}_box"] = valid_align_bbox
            y_true_final[f"{head_key}_dfl"] = valid_align_bbox
            y_true_final[f"{head_key}_class"] = align_cls

            sample_weights[f"{head_key}_box"] = (
                self.yolo_loss_weights.get(f"{head_key}_box", 1.0) * box_norm
            ) / cls_norm
            sample_weights[f"{head_key}_dfl"] = (
                self.yolo_loss_weights.get(f"{head_key}_dfl", 1.0) * box_norm
            ) / cls_norm
            sample_weights[f"{head_key}_class"] = (
                self.yolo_loss_weights.get(f"{head_key}_class", 1.0) / cls_norm
            )
        return super().compute_loss(
            x=x,
            y=y_true_final,
            y_pred=y_pred_final,
            sample_weight=sample_weights,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model": saving.serialize_keras_object(self.model),
                "head_keys": self.head_keys,
                "feature_map_shape": self.feature_map_shape,
                "input_size": self.input_size,
                "num_of_classes": self.num_of_classes,
                "iou": self.iou,
                "iou_factor": self.iou_factor,
                "cls_factor": self.cls_factor,
                "topk": self.topk,
                "reg_max": self.reg_max,
                "task": self.task,
                "mask_h": self.mask_h,
                "mask_w": self.mask_w,
            }
        )
        return config

    def get_compile_config(self):
        return super().get_compile_config()

    @classmethod
    def from_config(cls, config):
        return cls(**saving.deserialize_keras_object(config))

    def compile_from_config(self, config):
        config = saving.deserialize_keras_object(config)
        super().compile(**config)
        if hasattr(self, "optimizer") and self.built:
            # Create optimizer variables.
            self.optimizer.build(self.trainable_variables)

    def get_build_config(self):
        return {
            "input_shape": self.model.input_shape,
        }
