from typing import List, Tuple

from keras import KerasTensor, Model, ops


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
