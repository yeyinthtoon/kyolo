from keras import losses, ops, saving

from kyolo.utils.bounding_box_utils import calculate_iou


@saving.register_keras_serializable()
class BoxLoss(losses.Loss):
    def __init__(self, iou="ciou", **kwargs):
        super().__init__(**kwargs)
        self.iou = iou

    def call(self, y_true, y_pred):
        iou = calculate_iou(
            y_true, y_pred, self.dtype, metrics=self.iou, pairwise=False
        )
        iou_loss = 1.0 - iou
        return iou_loss

    def get_config(self):
        configs = super().get_config()
        configs.update({"iou": self.iou})
        return configs


@saving.register_keras_serializable()
class DFLLoss(losses.Loss):
    def __init__(self, anchor_norm, reg_max, **kwargs):
        super().__init__(**kwargs)
        self.anchor_norm = anchor_norm
        self.reg_max = reg_max

    def call(self, y_true, y_pred):
        valid_mask, y_true = ops.split(y_true, [1], axis=-1)
        left_target, right_target = ops.split(y_true, 2, axis=-1)
        target_dist = ops.concatenate(
            [(self.anchor_norm - left_target), (right_target - self.anchor_norm)],
            axis=-1,
        )
        target_dist = ops.clip(target_dist, 0.0, self.reg_max - 1.01)

        target_left, target_right = ops.floor(target_dist), ops.floor(target_dist) + 1
        weight_left, weight_right = (
            target_right - target_dist,
            target_dist - target_left,
        )
        # TODO: check correctness
        loss_left = losses.sparse_categorical_crossentropy(
            target_left, y_pred, from_logits=True
        )
        loss_right = losses.sparse_categorical_crossentropy(
            target_right, y_pred, from_logits=True
        )
        dfl_loss = loss_left * weight_left + loss_right * weight_right
        dfl_loss = ops.mean(dfl_loss, axis=-1)*ops.squeeze(valid_mask)
        return dfl_loss

    def get_config(self):
        configs = super().get_config()
        configs.update(
            {
                "anchor_norm": self.anchor_norm,
                "reg_max": self.reg_max,
            }
        )
        return configs


@saving.register_keras_serializable()
class BCELoss(losses.Loss):
    def __init__(self, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        bce = losses.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits, axis=[])
        return ops.sum(bce, axis=[1, 2])
    
    def get_config(self):
        configs = super().get_config()
        configs.update(
            {
                "from_logits": self.from_logits
            }
        )
        return configs 

# TODO: Update with new normalization style
@saving.register_keras_serializable()
class SegBCELoss(losses.Loss):
    def __init__(self, eps=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, y_true, y_pred):
        box_area, valid_mask, aligned_seg_mask = ops.split(y_true, [1, 2], axis=-1)
        seg_loss = losses.binary_crossentropy(
            aligned_seg_mask, y_pred, from_logits=False
        )
        seg_loss_unnormalize = ops.sum(
            seg_loss / (ops.squeeze(box_area) + self.eps), axis=-1
        )
        if self.reduction == "sum":
            return seg_loss_unnormalize / ops.maximum(ops.sum(valid_mask), 1)
        return seg_loss_unnormalize / ops.maximum(ops.sum(valid_mask, axis=(1, 2)), 1)
