from dataclasses import asdict
from functools import partial
from typing import Callable, Dict, List, Literal, Optional, Tuple

from keras import Model, ops, saving

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
        segmentation_loss_weight: int = 1,
        head_loss_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        head_loss_weights = {} if not head_loss_weights else head_loss_weights
        losses = {}
        loss_weights = {}

        for head_key in self.head_keys:
            head_loss_weight = head_loss_weights.get(head_key, 1.0)
            losses[f"{head_key}_box"] = box_loss(
                jit_compile=kwargs.get("jit_compile", False), iou=box_loss_iou
            )
            losses[f"{head_key}_class"] = classification_loss
            losses[f"{head_key}_dfl"] = dfl_loss(
                self.anchor_norm, self.reg_max, kwargs.get("jit_compile", False)
            )
            loss_weights[f"{head_key}_box"] = box_loss_weight * head_loss_weight
            loss_weights[f"{head_key}_class"] = (
                classification_loss_weight * head_loss_weight
            )
            loss_weights[f"{head_key}_dfl"] = dfl_loss_weight * head_loss_weight
            if self.task == "segmentation":
                if not segmentation_loss:
                    raise ValueError("missing segmentation loss")
                losses[f"{head_key}_segmentation"] = segmentation_loss
                loss_weights[f"{head_key}_segmentation"] = (
                    segmentation_loss_weight * head_loss_weight
                )

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

            align_cls, align_bbox, valid_mask, aligned_indices = (
                self.get_aligned_targets_detection(
                    cls,
                    boxes,
                    y["classes"],
                    y["bboxes"],
                    self.num_of_classes,
                    self.anchors,
                    self.dtype,
                )
            )

            align_bbox = align_bbox / self.scalers[None, ..., None]
            boxes = boxes / self.scalers[None, ..., None]

            y_pred_final[f"{head_key}_box"] = boxes
            y_pred_final[f"{head_key}_dfl"] = anchors
            y_pred_final[f"{head_key}_class"] = cls

            box_target = ops.concatenate(
                [align_bbox, valid_mask[..., None], align_cls], axis=-1
            )

            if self.task == "segmentation":
                mask_embs, protos = (
                    ops.cast(y_pred[head_key][3], self.dtype),
                    ops.cast(y_pred[head_key][4], self.dtype),
                )
                y_pred_final[f"{head_key}_masks"] = ops.einsum(
                    "bne,bhwe->bnhw", mask_embs, protos
                )

                aligned_indices_mask = ops.tile(
                    aligned_indices[:, :, 0, None, None],
                    (1, 1, self.mask_h, self.mask_w),
                )
                align_target_mask = ops.take_along_axis(
                    ops.transpose(y["masks"], (0, 3, 1, 2)),
                    aligned_indices_mask,
                    axis=1,
                )
                shapes = ops.shape(align_target_mask)
                batch = shapes[0]
                max_predict = shapes[1]
                align_target_mask = ops.reshape(
                    align_target_mask, (batch, max_predict, -1)
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
        config.update(
            {
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

    def compile_from_config(self, config):
        config = saving.deserialize_keras_object(config)
        super().compile(**config)
        if hasattr(self, "optimizer") and self.built:
            # Create optimizer variables.
            self.optimizer.build(self.trainable_variables)
