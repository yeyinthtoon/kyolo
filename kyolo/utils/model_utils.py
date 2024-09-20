from typing import List, Optional

from keras import KerasTensor, ops
from keras_cv.layers import MultiClassNonMaxSuppression

from kyolo.utils.bounding_box_utils import Vec2Box


class PostProccess:
    """
    Scale the predictions back and return prediction boxes after nms
    """

    def __init__(self, converter: Vec2Box, nms_cfg) -> None:
        self.converter = converter
        self.nms = nms_cfg

    def __call__(
        self, predict, rev_tensor: Optional[KerasTensor] = None
    ) -> List[KerasTensor]:
        prediction = self.converter(predict["MAIN"])
        pred_class, _, pred_bbox = prediction[:3]
        # TODO add support for older versions of yolo

        if rev_tensor is not None:
            pred_bbox = (pred_bbox - rev_tensor[:, None, 1:]) / rev_tensor[:, 0:1, None]

        nms = MultiClassNonMaxSuppression(
            "xyxy", True, confidence_threshold=self.nms.min_confidence
        )(pred_bbox, pred_class)
        pred_bbox_nms = ops.concatenate(
            [
                nms["classes"][:, :, None],
                nms["boxes"],
                nms["confidence"][:, :, None],
            ],
            axis=-1,
        )[:, : nms["num_detections"][0]]

        return pred_bbox_nms
