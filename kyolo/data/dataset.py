from functools import partial

import tensorflow as tf

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
    width = tf.cast(example["image/width"], "float32")
    height = tf.cast(example["image/height"], "float32")
    tf.ensure_shape(image, [None, None, 3])
    image.set_shape([None, None, 3])
    mask_raws = tf.sparse.to_dense(example["image/object/mask/binary"])
    xmins = tf.sparse.to_dense(example["image/object/bbox/xmin"]) * width
    ymins = tf.sparse.to_dense(example["image/object/bbox/ymin"]) * height
    xmaxs = tf.sparse.to_dense(example["image/object/bbox/xmax"]) * width
    ymaxs = tf.sparse.to_dense(example["image/object/bbox/ymax"]) * height
    classes = tf.sparse.to_dense(example["image/object/class/label"])
    bboxes = tf.stack([xmins, ymins, xmaxs, ymaxs], axis=-1)
    bboxes = tf.cast(bboxes, "float64")
    masks = tf.map_fn(decode_png_mask, mask_raws, fn_output_signature=tf.uint8)
    masks = tf.transpose(masks, (1, 2, 0))
    labels = {
        "classes": classes,
        "bboxes": bboxes,
    }
    if task == "segmentation":
        labels.update({"masks": masks})
    return image, labels


def pad_and_resize(image, labels, target_size=(640, 640)):
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
        raise ValueError(
            f"'image({image.get_shape().ndims})' must have either 3 or 4 dimensions."
        )
    scale_w = target_width / im_width
    scale_h = target_height / im_height
    scale = tf.math.minimum(scale_w, scale_h)
    new_width = tf.cast(tf.cast(im_width, "float64") * scale, "int32")
    new_height = tf.cast(tf.cast(im_height, "float64") * scale, "int32")

    padded_img = tf.image.resize_with_pad(
        image, target_height, target_width, method=tf.image.ResizeMethod.LANCZOS3
    )

    # process bbox
    pad_left = (target_width - new_width) / 2
    pad_top = (target_height - new_height) / 2

    x1, y1, x2, y2 = tf.split(labels["bboxes"], 4, axis=-1)

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
def decode_and_process_data(example, config, task):
    image, labels = decode_features(example, task)
    target_size = config.get("img_size", 640)
    image, labels = pad_and_resize(image, labels, (target_size, target_size))
    if task == "segmentation":
        image, labels = pad_and_resize_mask(image, labels, mask_ratio=1)

    return image, labels


@tf.function
def batched_data_process(images, labels, config):
    labels = labels.copy()
    # num_classes = config["num_classes"]
    default_pad_value = config.get("default_pad_value", 0)
    max_detection = config["max_detection"]
    mask_ratio = config["mask_ratio"]
    # seed = config["seed"]
    # images, labels = random_flip(images, labels, seed, "vertical", prob=0.25)
    # images, labels = random_flip(images, labels, seed, "horizontal", prob=0.25)
    if isinstance(images, tf.RaggedTensor):
        images = images.to_tensor(default_value=default_pad_value)
    images = tf.image.convert_image_dtype(images, tf.float32)

    classes = tf.cast(labels["classes"], tf.int32)
    # classes = tf.one_hot(labels["classes"], num_classes, dtype=tf.float32)
    labels["classes"] = classes.to_tensor(
        default_value=default_pad_value, shape=[None, max_detection]
    )[:,:,None]

    bboxes = tf.cast(labels["bboxes"], tf.float32)
    labels["bboxes"] = bboxes.to_tensor(
        default_value=default_pad_value, shape=[None, max_detection, 4]
    )
    if "masks" in labels:
        masks = tf.image.convert_image_dtype(labels["masks"], tf.float32)
        labels["masks"] = masks.to_tensor(
            default_value=default_pad_value,
            shape=[None, *masks.shape[1:3], max_detection],
        )
        _, labels = pad_and_resize_mask(images, labels, mask_ratio=mask_ratio)
    return images, labels


def build_tfrec_dataset(tfrec_files, config):
    dataset = tf.data.Dataset.from_tensor_slices(tfrec_files)

    dataset = dataset.shuffle(buffer_size=100, reshuffle_each_iteration=True)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=tf.data.AUTOTUNE).map(
            parse_example, num_parallel_calls=tf.data.AUTOTUNE
        ),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    dataset = dataset.shuffle(buffer_size=500, reshuffle_each_iteration=True)

    dataset = dataset.cache()
    decode_and_process_data_fn = partial(
        decode_and_process_data, config=config, task=config["task"]
    )
    dataset = dataset.map(
        decode_and_process_data_fn, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.ragged_batch(config["batch_size"])
    batched_data_process_fn = partial(batched_data_process, config=config)
    dataset = dataset.map(batched_data_process_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
