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


def decode_features(example):
    image_raw = example["image/encoded"]
    image = tf.image.decode_image(image_raw)
    width = tf.cast(example["image/width"], "float32")
    height = tf.cast(example["image/height"], "float32")
    tf.ensure_shape(image, [None, None, 3])
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
    return image, classes, bboxes, masks


def pad_and_resize(
    image, classes, boxes, masks, target_size=(640, 640), mask_ratio: int = 4
):
    target_height, target_width = target_size
    im_width = tf.shape(image)[1]
    im_height = tf.shape(image)[0]
    scale_w = target_width / im_width
    scale_h = target_height / im_height
    scale = tf.math.minimum(scale_w, scale_h)
    new_width = tf.cast(tf.cast(im_width, "float64") * scale, "int32")
    new_height = tf.cast(tf.cast(im_height, "float64") * scale, "int32")

    padded_img = tf.image.resize_with_pad(
        image, target_height, target_width, method=tf.image.ResizeMethod.LANCZOS3
    )
    padded_masks = tf.image.resize_with_pad(
        masks,
        target_height // mask_ratio,
        target_width // mask_ratio,
        method=tf.image.ResizeMethod.LANCZOS3,
    )

    pad_left = (target_width - new_width) / 2
    pad_top = (target_height - new_height) / 2

    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)

    x1 = x1 * scale + pad_left
    y1 = y1 * scale + pad_top
    x2 = x2 * scale + pad_left
    y2 = y2 * scale + pad_top

    boxes = tf.concat([x1, y1, x2, y2], axis=-1)

    return padded_img, classes, boxes, padded_masks


def preprocess_data(images, classes, bboxes, masks, max_labels=200, default_value=0):
    images = images.to_tensor(default_value=default_value)
    images = tf.image.convert_image_dtype(images, tf.float32)

    classes = classes.to_tensor(default_value=default_value, shape=[None, max_labels])
    bboxes = bboxes.to_tensor(default_value=default_value, shape=[None, max_labels, 4])
    masks = masks.to_tensor(
        default_value=default_value, shape=[None, *masks.shape[1:3], max_labels]
    )
    masks = tf.image.convert_image_dtype(masks, tf.float32)
    labels = {
        "classes": tf.cast(classes, tf.float32),
        "bboxes": tf.cast(bboxes, tf.float32),
        "masks": masks,
    }
    return images, labels


def build_tfrec_dataset(tfrec_files, batch_size):
    dataset = tf.data.TFRecordDataset(tfrec_files)
    dataset = dataset.map(parse_example)
    dataset = dataset.map(decode_features)
    dataset = dataset.map(pad_and_resize)
    dataset = dataset.ragged_batch(batch_size)
    dataset = dataset.map(preprocess_data)
    return dataset
