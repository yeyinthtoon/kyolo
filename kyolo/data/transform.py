import keras
import tensorflow as tf
from tensorflow import Tensor
from typing import Tuple, Dict


def random_hsvc(image: Tensor,
                v_max_delta: float = 0.2,
                h_max_delta: float = 0.2,
                s_min: float = 0,
                s_max: float = 2,
                c_min: float = 0,
                c_max: float = 2) -> Tensor:
    """
    Random hue, saturation, value, contrast augmentation.

    Args:
        image: Image, shape [H, W, C]
        v_max_delta: max delta value for brightness augmentation
        h_max_delta: max delta value for hue augmentation
        s_min: min saturation value
        s_max: max saturation value
        c_min: min contrast value
        c_max: max contrast value

    Returns:
        image: Image, shape [H, W, C]
    """

    image = tf.image.random_brightness(image, v_max_delta)
    image = tf.image.random_hue(image, h_max_delta)
    image = tf.image.random_saturation(image, s_min, s_max)
    image = tf.image.random_contrast(image, c_min, c_max)
    image = tf.clip_by_value(image, 0, 1)
    return image


def _get_kernel(kernel_size: Tuple, sigma):
    x = tf.cast(
        tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1),
        dtype=tf.float32,
    )
    blur_kernel = tf.exp(
        -tf.pow(x, 2.0)
        / (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0))
    )
    blur_kernel /= tf.reduce_sum(blur_kernel)
    return blur_kernel


def random_blur(image: Tensor,
                kernel_size: Tuple = (3, 3),
                min_sigma: int = 0,
                max_sigma: int = 10) -> Tensor:
    """
    Random gaussian blur augmentation.

    Args:
        image: Image, shape [H, W, C]
        kernel_size: Tuple of 2 integers representing gaussian kernel size
        min_sigma: Min value for sigma value
        max_sigma: Max value for sigma value

    Returns:
        image: Image, shape [H, W, C]
    """

    sigma = tf.random.uniform((1,), min_sigma, max_sigma)
    sigma = tf.math.maximum(sigma, keras.backend.epsilon())
    
    x, y = kernel_size
    kernel_x, kernel_y = _get_kernel(x, sigma), _get_kernel(y, sigma)
    kernel_x = tf.reshape(kernel_x, [1, x, 1, 1])
    kernel_y = tf.reshape(kernel_y, [y, 1, 1, 1])
    
    image = tf.expand_dims(image, axis=0)
    channels = image.shape[-1]

    kernel_x = tf.tile(kernel_x, [1, 1, channels, 1])
    kernel_y = tf.tile(kernel_y, [1, 1, channels, 1])
    
    image = tf.nn.depthwise_conv2d(image, kernel_x, strides=[1, 1, 1, 1], padding="SAME")
    image = tf.nn.depthwise_conv2d(image, kernel_y, strides=[1, 1, 1, 1], padding="SAME")
    return tf.squeeze(image, 0)


def random_flip(images: Tensor, labels: Dict[str,Tensor], seed: int=100, orientation: str = "vertical", prob: float = 0.5) -> Tuple[Tensor, Dict[str,Tensor]]:
    """
    Flip the image+bounding_boxes up/down or left/right randomly

    Args:
        images: Image [B, H, W, C] or [H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
        seed: Randomization seed
        orientation: To flip horizontally or vertically
        prob: Flip probability [0., 1.]

    Returns:
        images: Image [B, H, W, C] or [H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
    """

    boxes = labels["bboxes"]
    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=seed) >= prob:
        return images, labels

    if images.get_shape().ndims == 3:
        im_width = tf.shape(images)[1]
        im_height = tf.shape(images)[0]
        start_axis = 0
    elif images.get_shape().ndims == 4:
        im_width = tf.shape(images)[2]
        im_height = tf.shape(images)[1]
        start_axis = 1
    else:
      raise ValueError(f'\'image({images.get_shape().ndims})\' must have either 3 or 4 dimensions.')
    
    im_width = tf.cast(im_width, boxes.dtype)
    im_height = tf.cast(im_height, boxes.dtype)

    xmin, ymin, xmax, ymax = tf.split(boxes, 4, axis=-1)

    xmin = xmin / im_width
    ymin = ymin / im_height
    xmax = xmax / im_width
    ymax = ymax / im_height

    if orientation == "horizontal":
        images = tf.image.flip_left_right(images)
        if "masks" in labels:
            labels["masks"] = tf.reverse(labels["masks"], axis = [start_axis+1])
        new_xmin = 1 - xmax
        new_xmax = 1 - xmin
        new_ymin = ymin
        new_ymax = ymax
    elif orientation == "vertical":
        images = tf.image.flip_up_down(images)
        if "masks" in labels:
            labels["masks"] = tf.reverse(labels["masks"], axis = [start_axis])
        new_xmin = xmin
        new_xmax = xmax
        new_ymin = 1 - ymax
        new_ymax = 1 - ymin
    else:
        raise Exception("Unsupported orientation. Must be 'horizontal' or 'vertical'")

    xmin = new_xmin * im_width
    ymin = new_ymin * im_height
    xmax = new_xmax * im_width
    ymax = new_ymax * im_height

    labels["bboxes"] = tf.concat([xmin, ymin, xmax, ymax], axis=-1)
    return images, labels


def _filter_invalid_boxes(boxes: Tensor, cls: Tensor, min_area: float=1e-8) -> Tuple[Tensor, Tensor]:
    """
    Filter boxes with invalid points and
    filter boxes which are larger than min_area.
    (N >= M)

    Args:
        boxes: Set of bounding boxes, shape [N, 4]
        cls: Set of class labels, shape [N, C]
        min_area: Min area in float type to filter the boxes

    Returns:
        boxes: Set of bounding boxes, shape[M, 4]
        cls: Set of class labels, shape [M, C]
    """

    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (boxes_area > min_area)

    return boxes[valid_boxes], cls[valid_boxes]


def _filter_and_pad_invalid_boxes(boxes: Tensor, cls: Tensor, min_area: float=1e-8, default_pad_value: int=0) -> Tuple[Tensor, Tensor]:
    """
    Filter boxes with invalid points and
    filter boxes which are larger than min_area.
    Pad the return boxes and classes.

    Args:
        boxes: Set of bounding boxes, shape [B, N, 4]
        cls: Set of class labels, shape [B, N, C]
        min_area: Min area in float type to filter the boxes
        default_pad_value: Value to pad the output tensors

    Returns:
        boxes: Set of bounding boxes, shape[B, N, 4]
        cls: Set of class labels, shape [B, N, C]
    """

    num_class = tf.shape(cls)[-1]

    boxes_area = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])
    valids = (boxes[:, :, 2] > boxes[:, :, 0]) & (boxes[:, :, 3] > boxes[:, :, 1]) & (boxes_area > min_area)

    valid_boxes = tf.repeat(tf.expand_dims(valids, -1), 4, -1)
    valid_cls = tf.repeat(tf.expand_dims(valids, -1), num_class, -1)

    boxes = tf.where(valid_boxes, boxes, default_pad_value)
    cls = tf.where(valid_cls, cls, default_pad_value)

    return boxes, cls


def random_crop(images: Tensor, labels: Dict[str, Tensor], crop_height: int, crop_width: int, prob: float, seed: int) -> Tuple[Tensor, Dict[str,Tensor]]:
    """
    Crop the image+bounding_boxes+labels into (crop_height, crop_width) size randomly.
    Output image size will be the same as input image size.

    Args:
        images: Image, shape [H, W, C]
        labels: Dict of "classes", "bboxes" and "masks"
        crop_height: Crop area height
        crop_width: Crop area width
        prob: Crop probability [0., 1.]
        seed: Randomization seed

    Returns:
        images: Image, shape [H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
    """

    boxes = labels.get('bboxes', None)
    cls = labels.get('classes', None)

    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=seed) >= prob:
        return images, labels

    original_height = tf.shape(images)[0]
    original_width = tf.shape(images)[1]

    original_height = tf.cast(original_height, boxes.dtype)
    original_width = tf.cast(original_width, boxes.dtype)
    crop_height = tf.cast(crop_height, boxes.dtype)
    crop_width = tf.cast(crop_width, boxes.dtype)

    if crop_width > original_width or crop_height > original_height:
        return images, labels

    left = tf.random.uniform([], 0, original_width - crop_width, dtype=boxes.dtype)
    top = tf.random.uniform([], 0, original_height - crop_height, dtype=boxes.dtype)

    # augment the images
    output_image = tf.image.crop_and_resize(
        images[None],
        tf.stack([
            top/original_height,
            left/original_width,
            (top+crop_height)/original_height,
            (left+crop_width)/original_width], axis=-1)[None],
        tf.range(1),
        [original_height, original_width],
        method="bilinear",
    )

    # augment the masks
    if "masks" in labels:
        masks = labels['masks']

        masks = tf.image.crop_and_resize(
            masks[None],
            tf.stack([
                top/original_height,
                left/original_width,
                (top+crop_height)/original_height,
                (left+crop_width)/original_width], axis=-1)[None],
            tf.range(1),
            [original_height, original_width],
            method="bilinear",
        )
        labels['masks'] = tf.cast(masks[0], tf.uint8)

    # augment the boxes and cls
    left = tf.reshape(left, [-1, 1])
    top = tf.reshape(top, [-1, 1])

    # transform the boxes
    xmin, ymin, xmax, ymax = tf.split(boxes, 4, axis=-1)
    xmin = tf.clip_by_value(xmin - left, 0, crop_width) * (original_width/crop_width)
    ymin = tf.clip_by_value(ymin - top, 0, crop_height) * (original_height/crop_height)
    xmax = tf.clip_by_value(xmax - left, 0, crop_width) * (original_width/crop_width)
    ymax = tf.clip_by_value(ymax - top, 0, crop_height) * (original_height/crop_height)
    boxes = tf.concat([xmin, ymin, xmax, ymax], axis=-1)

    # filter invalid boxes
    cls = tf.expand_dims(cls, -1)
    boxes, cls = _filter_invalid_boxes(boxes, cls)

    labels['bboxes'] = boxes
    labels['classes'] = tf.squeeze(cls, -1)

    return output_image[0], labels


def mosaic(images: Tensor, labels: Dict[str,Tensor], default_pad_value: int, generator: tf.random.Generator) -> Tuple[Tensor, Dict[str,Tensor]]:
    """
    Tile 4 random images and random crop from a batch of data

    Args:
        images: Image [B, H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
        default_pad_value: Constant value to pad the boxes and classes
        generator: TF random generator

    Returns:
        images: Image [B, H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
    """

    boxes = labels['bboxes']
    cls = labels['classes']

    batch_size = tf.shape(images)[0]
    input_height = tf.shape(images)[1]
    channels = tf.shape(images)[3]

    image_pairs = generator.uniform(
        (batch_size, 3),
        minval=0,
        maxval=batch_size,
        dtype=tf.int32
    )

    image_pairs = tf.concat(
        [tf.range(batch_size)[:, None], image_pairs],
        axis=-1
    )

    # augment images
    mosaic_images = tf.gather(images, image_pairs)

    tops = tf.concat([mosaic_images[:, 0], mosaic_images[:, 1]], axis=2)
    bottoms = tf.concat([mosaic_images[:, 2], mosaic_images[:, 3]], axis=2)
    output_images = tf.concat([tops, bottoms], axis=1)

    imgsz = input_height
    border = (-imgsz // 2, -imgsz // 2)

    yc = tf.random.uniform([], -border[0], 2 * imgsz + border[0], dtype=tf.int32)
    xc = tf.random.uniform([], -border[1], 2 * imgsz + border[1], dtype=tf.int32)

    left = xc - imgsz//2
    top = yc - imgsz//2

    left = tf.cast(left, tf.int32)
    top = tf.cast(top, tf.int32)

    output_images = tf.slice(output_images, \
                             [0, top, left, 0], \
                             [batch_size, imgsz, imgsz, channels])

    # augment masks
    if "masks" in labels:
        masks = labels['masks']

        mask_height = tf.shape(masks)[1]
        mask_width = tf.shape(masks)[2]
        mask_count = tf.shape(masks)[3]

        mask_ratio = input_height // mask_height

        mosaic_masks = tf.gather(masks, image_pairs)
        mosaic_masks = tf.split(mosaic_masks, 4, axis=1)

        output_masks = tf.concat([
            tf.image.pad_to_bounding_box(
                tf.squeeze(mosaic_masks[0], 1),
                0,
                0,
                mask_height*2,
                mask_width*2
            ),
            tf.image.pad_to_bounding_box(
                tf.squeeze(mosaic_masks[1], 1),
                0,
                mask_width,
                mask_height*2,
                mask_width*2
            ),
            tf.image.pad_to_bounding_box(
                tf.squeeze(mosaic_masks[2], 1),
                mask_height,
                0,
                mask_height*2,
                mask_width*2
            ),
            tf.image.pad_to_bounding_box(
                tf.squeeze(mosaic_masks[3], 1),
                mask_height,
                mask_width,
                mask_height*2,
                mask_width*2
            )
        ], axis=-1)

        output_masks = tf.slice(output_masks, \
                             [0, int(top/mask_ratio), int(left/mask_ratio), 0], \
                             [batch_size, mask_height, mask_width, mask_count*4])

        labels["masks"] = tf.image.convert_image_dtype(output_masks, tf.float32)

    # augment boxes and classes
    batch_size = tf.shape(boxes)[0]
    box_count = tf.shape(boxes)[1]

    imgsz = tf.cast(imgsz, boxes.dtype)
    left = tf.cast(left, boxes.dtype)
    top = tf.cast(top, boxes.dtype)

    mosaic_boxes = tf.gather(boxes, image_pairs)
    mosaic_boxes = tf.split(mosaic_boxes, 4, axis=1)

    mosaic_cls = tf.gather(cls, image_pairs)
    mosaic_cls = tf.split(mosaic_cls, 4, axis=1)
    cls = tf.reshape(mosaic_cls, (batch_size, 4*box_count, -1))

    transform_matrix_1 = tf.concat([
        tf.fill((batch_size, box_count, 1), imgsz),
        tf.fill((batch_size, box_count, 1), 0.),
        tf.fill((batch_size, box_count, 1), imgsz),
        tf.fill((batch_size, box_count, 1), 0.),
    ], axis=-1)

    transform_matrix_2 = tf.concat([
        tf.fill((batch_size, box_count, 1), 0.),
        tf.fill((batch_size, box_count, 1), imgsz),
        tf.fill((batch_size, box_count, 1), 0.),
        tf.fill((batch_size, box_count, 1), imgsz),
    ], axis=-1)

    transform_matrix_3 = tf.concat([
        tf.fill((batch_size, box_count, 1), imgsz),
        tf.fill((batch_size, box_count, 1), imgsz),
        tf.fill((batch_size, box_count, 1), imgsz),
        tf.fill((batch_size, box_count, 1), imgsz),
    ], axis=-1)

    boxes = tf.concat([
        tf.squeeze(mosaic_boxes[0], axis=1),
        tf.squeeze(mosaic_boxes[1], axis=1) + transform_matrix_1,
        tf.squeeze(mosaic_boxes[2], axis=1) + transform_matrix_2,
        tf.squeeze(mosaic_boxes[3], axis=1) + transform_matrix_3,
    ], axis=1)

    left = tf.reshape(left, [-1, 1])
    top = tf.reshape(top, [-1, 1])

    xmin, ymin, xmax, ymax = tf.split(boxes, 4, axis=-1)
    xmin = tf.clip_by_value(xmin - left, 0, imgsz)
    ymin = tf.clip_by_value(ymin - top, 0, imgsz)
    xmax = tf.clip_by_value(xmax - left, 0, imgsz)
    ymax = tf.clip_by_value(ymax - top, 0, imgsz)
    boxes = tf.concat([xmin, ymin, xmax, ymax], axis=-1)

    # filter invalid boxes
    boxes, cls = _filter_and_pad_invalid_boxes(boxes, cls, default_pad_value=default_pad_value)

    labels["bboxes"] = boxes
    labels["classes"] = cls

    return output_images, labels