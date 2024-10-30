import inspect
from functools import partial
from typing import Callable, Dict

import tensorflow as tf
from keras import backend
from kyolo.data import transform
from kyolo.data.transform import TRANSFORMS_REGISTRY

feature_description = {
    "image/height": tf.io.FixedLenFeature([], tf.int64),
    "image/width": tf.io.FixedLenFeature([], tf.int64),
    "image/filename": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "image/encoded": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "image/format": tf.io.FixedLenFeature([], tf.string, default_value="jpeg"),
    "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    "image/object/class/text": tf.io.VarLenFeature(tf.string),
    "image/object/class/label": tf.io.VarLenFeature(tf.int64),
    "image/object/mask/binary": tf.io.VarLenFeature(tf.string),
    "image/object/mask/polygon": tf.io.VarLenFeature(tf.float32),
}


BATCHED_PROCESSING_ORDERED = [
    "random_flip_vertical",
    "random_flip_horizontal",
    "labels_padding_process",
    "mosaic",
    "filter_data_process",
    "random_perspective"
]


def parse_example(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)


def decode_png_mask(png_bytes):
    mask = tf.squeeze(tf.io.decode_png(png_bytes, channels=1, dtype=tf.uint8), axis=-1)
    tf.ensure_shape(mask, [None, None])
    return mask


def decode_features(example, task):
    image_raw = example["image/encoded"]
    image = tf.image.decode_image(image_raw)
    width = tf.cast(example["image/width"],"float32")
    height = tf.cast(example["image/height"],"float32")
    tf.ensure_shape(image,[None, None, 3])
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    mask_raws = tf.sparse.to_dense(example["image/object/mask/binary"])
    xmins = tf.sparse.to_dense(example["image/object/bbox/xmin"]) * width
    ymins = tf.sparse.to_dense(example["image/object/bbox/ymin"]) * height
    xmaxs = tf.sparse.to_dense(example["image/object/bbox/xmax"]) * width
    ymaxs = tf.sparse.to_dense(example["image/object/bbox/ymax"]) * height
    classes = tf.sparse.to_dense(example["image/object/class/label"])
    bboxes = tf.stack([xmins,ymins,xmaxs,ymaxs],axis=-1)
    bboxes = tf.cast(bboxes, "float32")
    masks = tf.map_fn(decode_png_mask,mask_raws, fn_output_signature=tf.uint8)
    masks = tf.transpose(masks,(1,2,0))

    labels = {
        "classes": classes,
        "bboxes": bboxes
    }

    if task == "segmentation":
        labels.update({"masks": masks})

    return image, labels


def pad_and_resize(image, labels, target_size=(640,640)):
    target_height, target_width = target_size
    if image.get_shape().ndims == 3:
        im_width = tf.shape(image)[1]
        im_height = tf.shape(image)[0]
    elif image.get_shape().ndims == 4:
        im_width = tf.shape(image)[2]
        im_height = tf.shape(image)[1]
    else:
        print(image.get_shape())
        print(image.get_shape().ndims)
        raise ValueError(f'\'image({image.get_shape().ndims})\' must have either 3 or 4 dimensions.')

    boxes = labels["bboxes"]

    scale_w = target_width / im_width
    scale_h = target_height / im_height
    scale = tf.math.minimum(scale_w, scale_h)
    scale = tf.cast(scale, boxes.dtype)
    new_width = tf.cast(tf.cast(im_width, scale.dtype) * scale, "int32")
    new_height = tf.cast(tf.cast(im_height, scale.dtype) * scale, "int32")

    padded_img = tf.image.resize_with_pad(
        image,
        target_height,
        target_width,
        method=tf.image.ResizeMethod.LANCZOS3
    )

    # process bbox
    pad_left = tf.cast((target_width - new_width) / 2, boxes.dtype)
    pad_top = tf.cast((target_height - new_height) / 2, boxes.dtype)

    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)

    x1 = x1 * scale + pad_left
    y1 = y1 * scale + pad_top
    x2 = x2 * scale + pad_left
    y2 = y2 * scale + pad_top

    labels["bboxes"] = tf.concat([x1, y1, x2, y2], axis=-1)

    return padded_img, labels


def pad_and_resize_mask(image, labels, mask_ratio: int = 4):
    if image.get_shape().ndims == 3:
        target_width = tf.shape(image)[1]
        target_height = tf.shape(image)[0]
    elif image.get_shape().ndims == 4:
        target_width = tf.shape(image)[2]
        target_height = tf.shape(image)[1]
    else:
        raise ValueError(
            f"'image({image.get_shape().ndims})' must have either 3 or 4 dimensions."
        )

    # process mask
    labels["masks"] = tf.image.resize_with_pad(
        labels["masks"],
        target_height // mask_ratio,
        target_width // mask_ratio,
        method=tf.image.ResizeMethod.LANCZOS3,
    )

    return image, labels


@tf.function
def decode_and_process_data(example, config, task, mode):
    image, labels = decode_features(example, task)
    target_size = config.get("img_size", 640)
    if mode == "train":
        transforms = config.get("transforms", {})
        for fname, kwargs in transforms.items():
            transform_func = TRANSFORMS_REGISTRY[fname]
            if "labels" in set(inspect.signature(transform_func).parameters.keys()):
                image, labels = transform_func(image, labels, **kwargs)
            else:
                image = transform_func(image, **kwargs)
    image, labels = pad_and_resize(image, labels, (target_size, target_size))
    if task == "segmentation":
        image, labels = pad_and_resize_mask(image, labels, mask_ratio=1)

    return image, labels


@tf.function
def labels_padding_process(images, labels, img_size, num_classes, default_pad_value, max_detection, mask_ratio):

    if isinstance(images, tf.RaggedTensor):
        images = images.to_tensor(default_value=default_pad_value)
    images = tf.image.convert_image_dtype(images, tf.float32)

    labels = labels.copy()

    classes = tf.cast(labels["classes"], tf.int32)
    # classes = tf.one_hot(
    #     labels["classes"],
    #     num_classes,
    #     dtype=tf.float32
    # )
    labels["classes"] = classes.to_tensor(
            default_value=default_pad_value,
            shape=[None, max_detection]
        )[:, :, None]

    bboxes = tf.cast(labels["bboxes"],tf.float32)
    labels["bboxes"] = bboxes.to_tensor(
            default_value=default_pad_value,
            shape=[None, max_detection, 4]
        )

    if "masks" in labels:
        masks = tf.image.convert_image_dtype(labels["masks"], tf.float32)
        labels["masks"] = masks.to_tensor(
                default_value=default_pad_value,
                shape=[None, *masks.shape[1:3], max_detection]
            )
        _, labels = pad_and_resize_mask(images, labels, mask_ratio=mask_ratio)

    return images, labels


@tf.function
def filter_data_process(images, labels, default_pad_value, max_detection):
    labels = labels.copy()

    boxes = labels.get("bboxes", None)
    classes = labels.get("classes", None)
    masks = labels.get("masks", None)

    boxes, classes, masks = transform.filter_and_pad_invalid_boxes(
                                boxes,
                                classes,
                                masks,
                                max_detection=max_detection,
                                default_pad_value=default_pad_value)
    
    labels["bboxes"] = boxes
    labels["classes"] = classes

    if not masks == None:
        labels["masks"] = masks

    return images, labels


def build_tfrec_dataset(tfrec_files, config, task, mode="train", drop_remainder=True):
    img_size = config.get("img_size", 640)
    num_classes = config.get("num_classes", 80)
    max_detection = config.get("max_detection", 100)
    mask_ratio = config.get("mask_ratio", 1)
    seed = config.get("seed", 62)
    batch_size = config.get("batch_size", 4)
    default_pad_value = config.get("default_pad_value", 0)
    mosaic = config.get("mosaic", False)
    
    generator = tf.random.Generator.from_seed(seed)
    
    dataset = tf.data.Dataset.from_tensor_slices(tfrec_files)
    if mode == "train":
        dataset = dataset.shuffle(buffer_size=100, reshuffle_each_iteration=True)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=tf.data.AUTOTUNE).map(
            parse_example, num_parallel_calls=tf.data.AUTOTUNE
        ),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    if config.get("cache", False):
        dataset = dataset.cache()

    if mode == "train":
        dataset = dataset.shuffle(buffer_size=500, reshuffle_each_iteration=True)
    decode_and_process_data_fn = partial(
        decode_and_process_data, config=config, task=task, mode=mode
    )
    dataset = dataset.map(
        decode_and_process_data_fn, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.ragged_batch(batch_size, drop_remainder=drop_remainder)
    
    batched_transforms = config.get("batched_transforms", {})
    
    for fname in BATCHED_PROCESSING_ORDERED:
        # data processing functions
        
        if fname == "filter_data_process":
            if mode == "train" and \
            ("mosaic" in batched_transforms or \
             "mixup" in batched_transforms):
                partial_fn = partial(
                                filter_data_process,
                                default_pad_value=default_pad_value,
                                max_detection=max_detection
                            )
                dataset = dataset.unbatch()
                dataset = dataset.map(partial_fn, num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.batch(batch_size)
            
        if fname == "labels_padding_process":
            partial_fn = partial(
                            labels_padding_process,
                            img_size=img_size,
                            num_classes=num_classes,
                            default_pad_value=default_pad_value,
                            max_detection=max_detection,
                            mask_ratio=mask_ratio
                        )
            dataset = dataset.map(partial_fn, num_parallel_calls=tf.data.AUTOTUNE)

        # data transformation functions
        
        if not fname in batched_transforms:
            continue
        
        kwargs = batched_transforms[fname]
        
        if mode == "train":
            if fname in TRANSFORMS_REGISTRY:
                transform_fn = TRANSFORMS_REGISTRY[fname]
                
                if "generator" in set(inspect.signature(transform_fn).parameters.keys()):
                    kwargs["generator"] = generator

                partial_fn = partial(
                        transform_fn,
                        **kwargs
                    )
                dataset = dataset.map(partial_fn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
