from keras import callbacks, ops
from keras_cv.layers import NonMaxSuppression
from keras_cv.src import bounding_box
from keras_cv.src.metrics.coco import compute_pycoco_metrics
from keras_cv.src.models.object_detection.__internal__ import unpack_input
from keras_cv.src.utils.conditional_imports import assert_pycocotools_installed


class PyCOCOCallback(callbacks.Callback):
    def __init__(
        self,
        validation_data,
        bounding_box_format,
        pred_key="main",
        nms_conf=0.5,
        nms_iou=0.3,
        max_detection=100,
        cache=True,
        **kwargs
    ):
        """Creates a callback to evaluate PyCOCO metrics on a validation
        dataset.

        Args:
            validation_data: a tf.data.Dataset containing validation data.
                Entries should have the form ```(images, {"boxes": boxes,
                "classes": classes})```.
            bounding_box_format: the KerasCV bounding box format used in the
                validation dataset (e.g. "xywh")
            cache: whether the callback should cache the dataset between
                iterations. Note that if the validation dataset has shuffling of
                any kind (e.g. from `shuffle_files=True` in a call to TFDS).
                Load or a call to tf.data.Dataset.shuffle() with
                `reshuffle_each_iteration=True`), you **must** cache the dataset
                to preserve iteration order. This will store your entire dataset
                in main memory, so for large datasets consider avoiding shuffle
                operations and passing `cache=False`.
        """
        assert_pycocotools_installed("PyCOCOCallback")
        self.val_data = validation_data
        if cache:
            # We cache the dataset to preserve a consistent iteration order.
            self.val_data = self.val_data.cache()
        self.bounding_box_format = bounding_box_format
        self.nms_conf = nms_conf
        self.nms_iou = nms_iou
        self.pred_key = pred_key
        self.nms = NonMaxSuppression(
            "xyxy",
            True,
            confidence_threshold=nms_conf,
            iou_threshold=nms_iou,
            max_detections=max_detection,
        )
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def images_only(data, maybe_boxes=None):
            if maybe_boxes is None:
                images, boxes = unpack_input(data)
            else:
                images = data
            return images

        def boxes_only(data, maybe_boxes=None):
            if maybe_boxes is None:
                images, boxes = unpack_input(data)
            else:
                boxes = maybe_boxes
            return boxes

        images_only_ds = self.val_data.map(images_only)
        y_pred_ = self.model.predict(images_only_ds)[self.pred_key]
        y_pred = self.nms(y_pred_[2], y_pred_[0])
        box_pred = y_pred["boxes"]
        cls_pred = ops.convert_to_numpy(y_pred["classes"])
        confidence_pred = ops.convert_to_numpy(y_pred["confidence"])
        valid_det = ops.convert_to_numpy(y_pred["num_detections"])

        gt = [boxes for boxes in self.val_data.map(boxes_only)]
        gt_boxes = ops.concatenate(
            [ops.convert_to_numpy(boxes["bboxes"]) for boxes in gt],
            axis=0,
        )
        gt_classes = ops.squeeze(
            ops.concatenate(
                [ops.convert_to_numpy(boxes["classes"]) for boxes in gt],
                axis=0,
            ),
            -1,
        )

        first_image_batch = next(iter(images_only_ds))
        height = first_image_batch.shape[1]
        width = first_image_batch.shape[2]
        total_images = gt_boxes.shape[0]

        gt_boxes = bounding_box.convert_format(
            gt_boxes, source=self.bounding_box_format, target="yxyx"
        )

        source_ids = np.char.mod("%d", np.linspace(1, total_images, total_images))
        num_detections = ops.sum(ops.cast(gt_classes > 0, "int32"), axis=-1)

        ground_truth = {
            "source_id": [source_ids],
            "height": [
                ops.convert_to_numpy(ops.tile(ops.array([height]), [total_images]))
            ],
            "width": [
                ops.convert_to_numpy(ops.tile(ops.array([width]), [total_images]))
            ],
            "num_detections": [ops.convert_to_numpy(num_detections)],
            "boxes": [ops.convert_to_numpy(gt_boxes)],
            "classes": [ops.convert_to_numpy(gt_classes)],
        }

        box_pred = bounding_box.convert_format(
            box_pred, source=self.bounding_box_format, target="yxyx"
        )

        predictions = {
            "source_id": [source_ids],
            "detection_boxes": [ops.convert_to_numpy(box_pred)],
            "detection_classes": [cls_pred],
            "detection_scores": [confidence_pred],
            "num_detections": [valid_det],
        }

        metrics = compute_pycoco_metrics(ground_truth, predictions)
        # Mark these as validation metrics by prepending a val_ prefix
        metrics = {"val_" + name: val for name, val in metrics.items()}

        logs.update(metrics)
