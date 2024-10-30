from math import pi
from typing import Callable, Dict, Tuple

import keras
import tensorflow as tf
from tensorflow import Tensor

TRANSFORMS_REGISTRY: Dict[str, Callable] = {}


def register_transforms(transforms_fn) -> Callable:
    """decorators to register data transformations"""
    if transforms_fn.__name__ in TRANSFORMS_REGISTRY:
        raise ValueError(
            f"Can't register same function twice. {transforms_fn.__name__}"
        )
    TRANSFORMS_REGISTRY[transforms_fn.__name__] = transforms_fn
    return transforms_fn


def _get_kernel(kernel_size: Tuple, sigma):
    x = tf.cast(
        tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1),
        dtype=tf.float32,
    )
    blur_kernel = tf.exp(
        -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0))
    )
    blur_kernel /= tf.reduce_sum(blur_kernel)
    return blur_kernel


def _filter_invalid_boxes(
    boxes: Tensor, cls: Tensor, min_area: float = 1e-8
) -> Tuple[Tensor, Tensor]:
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
    valid_boxes = (
        (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
        & (boxes_area > min_area)
    )

    return boxes[valid_boxes], cls[valid_boxes]


def _gather_scattered_boxes(boxes: Tensor, cls: Tensor, masks: Tensor=None, max_detection: int=200, box_pad_value: Tensor=None, cls_pad_value: Tensor=None, mask_pad_value: Tensor=None) -> Tensor:

    indices = tf.squeeze(tf.where(tf.reduce_max(boxes, -1) != 0), -1)

    box_gathered = tf.gather(boxes, indices, axis=0)
    cls_gathered = tf.gather(cls, indices, axis=0)
    
    if max_detection-tf.shape(box_gathered)[0] > 0:
        box_paddings = tf.fill((max_detection-tf.shape(box_gathered)[0], 4), box_pad_value)
        box_gathered = tf.concat([
                box_gathered,
                box_paddings
            ], axis=0)

        cls_paddings = tf.fill((max_detection-tf.shape(cls_gathered)[0], 1), cls_pad_value)
        cls_gathered = tf.concat([
                cls_gathered,
                cls_paddings
            ], axis=0)
        
    boxes = box_gathered[:max_detection]
    cls = cls_gathered[:max_detection]

    if not masks == None:
        mask_h = tf.shape(masks)[0]
        mask_w = tf.shape(masks)[1]

        mask_gathered = tf.gather(masks, indices, batch_dims=0, axis=-1)
        if max_detection-tf.shape(mask_gathered)[-1] > 0:
            mask_paddings = tf.fill((mask_h, mask_w, max_detection-tf.shape(mask_gathered)[-1]), mask_pad_value)
            mask_gathered = tf.concat([
                mask_gathered,
                mask_paddings
            ], axis=-1)
        masks = mask_gathered[:, :, :max_detection]

    return boxes, cls, masks


def filter_and_pad_invalid_boxes(boxes: Tensor, cls: Tensor, masks: Tensor=None, min_area: float=1e-8, max_detection: int=200, default_pad_value: int=0) -> Tuple[Tensor, Tensor]:
    """
    Filter boxes with invalid points and
    filter boxes which are larger than min_area.
    (N >= M)

    Args:
        boxes: Set of bounding boxes, shape [B, N, 4]
        cls: Set of class labels, shape [B, N, C]
        min_area: Min area in float type to filter the boxes
        default_pad_value: Value to pad the output tensors

    Returns:
        boxes: Set of bounding boxes, shape [B, M, 4]
        cls: Set of class labels, shape [B, M, C]
    """

    num_class = tf.shape(cls)[-1]

    box_pad_value = tf.convert_to_tensor(default_pad_value, boxes.dtype)
    cls_pad_value = tf.convert_to_tensor(default_pad_value, cls.dtype)
    mask_pad_value = None

    if not masks == None:
        mask_pad_value = tf.convert_to_tensor(default_pad_value, masks.dtype)

    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    valids = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (boxes_area > min_area)

    valid_boxes = tf.repeat(tf.expand_dims(valids, -1), 4, -1)
    valid_cls = tf.repeat(tf.expand_dims(valids, -1), num_class, -1)

    boxes = tf.where(valid_boxes, boxes, box_pad_value)
    cls = tf.where(valid_cls, cls, cls_pad_value)

    boxes, cls, masks = _gather_scattered_boxes(boxes, cls, masks, max_detection, box_pad_value, cls_pad_value, mask_pad_value)

    return boxes, cls, masks


def _get_rotation_matrix(angle, img_height, img_width):

    img_height = tf.cast(img_height, tf.float32)
    img_width = tf.cast(img_width, tf.float32)

    x0, y0 = (img_height-1)/2.0, (img_width-1)/2.0
    x_offset = x0*(1-tf.cos(angle))+ y0*tf.sin(angle)
    y_offset = y0*(1-tf.cos(angle))- x0*tf.sin(angle)

    return tf.convert_to_tensor([
                [tf.cos(angle), -tf.sin(angle), x_offset],
                [tf.sin(angle), tf.cos(angle), y_offset],
                [0., 0., 1.]
            ])


@register_transforms
def random_hsvc(
    image: Tensor,
    v_max_delta: float = 0.2,
    h_max_delta: float = 0.2,
    s_min: float = 0,
    s_max: float = 2,
    c_min: float = 0,
    c_max: float = 2,
) -> Tensor:
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


@register_transforms
def random_blur(
    image: Tensor, kernel_size: Tuple = (3, 3), min_sigma: int = 0, max_sigma: int = 10
) -> Tensor:
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

    image = tf.nn.depthwise_conv2d(
        image, kernel_x, strides=[1, 1, 1, 1], padding="SAME"
    )
    image = tf.nn.depthwise_conv2d(
        image, kernel_y, strides=[1, 1, 1, 1], padding="SAME"
    )
    return tf.squeeze(image, 0)


@register_transforms
def random_crop(
    images: Tensor,
    labels: Dict[str, Tensor],
    min_crop_ratio: float,
    max_crop_ratio: float,
    prob: float,
    seed: int,
) -> Tuple[Tensor, Dict[str, Tensor]]:
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

    boxes = labels.get("bboxes", None)
    cls = labels.get("classes", None)
    cls_dtype = cls.dtype
    masks = labels.get("masks", None)

    original_height = tf.shape(images)[0]
    original_width = tf.shape(images)[1]

    original_height = tf.cast(original_height, boxes.dtype)
    original_width = tf.cast(original_width, boxes.dtype)
    crop_height = tf.random.uniform(
        [],
        original_height * min_crop_ratio,
        original_height * max_crop_ratio,
        dtype=boxes.dtype,
    )
    crop_width = tf.random.uniform(
        [],
        original_width * min_crop_ratio,
        original_width * max_crop_ratio,
        dtype=boxes.dtype,
    )

    left = tf.random.uniform([], 0, original_width - crop_width, dtype=boxes.dtype)
    top = tf.random.uniform([], 0, original_height - crop_height, dtype=boxes.dtype)
    crop_box = tf.cast(
        tf.stack(
            [
                top / original_height,
                left / original_width,
                (top + crop_height) / original_height,
                (left + crop_width) / original_width,
            ],
            axis=-1,
        ),
        tf.float32,
    )[None]

    # augment the masks
    if not masks == None:
        masks = labels["masks"]

        masks = tf.image.crop_and_resize(
            masks[None],
            crop_box,
            tf.range(1),
            [original_height, original_width],
            method="bilinear",
        )
        labels["masks"] = tf.cast(masks[0], tf.uint8)

    if (
        tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=seed)
        >= prob
    ):
        return images, labels

    # augment the images
    output_image = tf.image.crop_and_resize(
        images[None],
        crop_box,
        tf.range(1),
        [original_height, original_width],
        method="bilinear",
    )[0]

    # augment the boxes and cls
    left = tf.reshape(left, [-1, 1])
    top = tf.reshape(top, [-1, 1])

    # transform the boxes
    xmin, ymin, xmax, ymax = tf.split(boxes, 4, axis=-1)
    xmin = tf.clip_by_value(xmin - left, 0, crop_width) * (original_width / crop_width)
    ymin = tf.clip_by_value(ymin - top, 0, crop_height) * (
        original_height / crop_height
    )
    xmax = tf.clip_by_value(xmax - left, 0, crop_width) * (original_width / crop_width)
    ymax = tf.clip_by_value(ymax - top, 0, crop_height) * (
        original_height / crop_height
    )
    boxes = tf.concat([xmin, ymin, xmax, ymax], axis=-1)

    # filter invalid boxes
    cls = tf.expand_dims(cls, -1)
    boxes, cls = _filter_invalid_boxes(boxes, cls)

    labels["bboxes"] = boxes
    labels["classes"] = tf.cast(tf.squeeze(cls, -1), cls_dtype)

    return output_image, labels


@register_transforms
@tf.function
def random_flip_vertical(images: Tensor, labels: Dict[str,Tensor], seed: int=100, prob: float = 0.5) -> Tuple[Tensor, Dict[str,Tensor]]:
    """
    Flip the image+bounding_boxes up/down or left/right randomly

    Args:
        images: Image [B, H, W, C] or [H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
        seed: Randomization seed
        prob: Flip probability [0., 1.]

    Returns:
        images: Image [B, H, W, C] or [H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
    """

    labels = labels.copy()

    boxes = labels.get("bboxes", None)
    masks = labels.get("masks", None)

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

    if not masks == None:
        labels["masks"] = tf.reverse(masks, axis = [start_axis])

    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=seed) >= prob:
        return images, labels

    im_width = tf.cast(im_width, boxes.dtype)
    im_height = tf.cast(im_height, boxes.dtype)

    xmin, ymin, xmax, ymax = tf.split(boxes, 4, axis=-1)

    xmin = xmin / im_width
    ymin = ymin / im_height
    xmax = xmax / im_width
    ymax = ymax / im_height

    images = tf.image.flip_up_down(images)
    new_xmin = xmin
    new_xmax = xmax
    new_ymin = 1 - ymax
    new_ymax = 1 - ymin

    xmin = new_xmin * im_width
    ymin = new_ymin * im_height
    xmax = new_xmax * im_width
    ymax = new_ymax * im_height

    labels["bboxes"] = tf.concat([xmin, ymin, xmax, ymax], axis=-1)
    return images, labels


@register_transforms
@tf.function
def random_flip_horizontal(images: Tensor, labels: Dict[str,Tensor], seed: int=100, prob: float = 0.5) -> Tuple[Tensor, Dict[str,Tensor]]:
    """
    Flip the image+bounding_boxes up/down or left/right randomly

    Args:
        images: Image [B, H, W, C] or [H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
        seed: Randomization seed
        prob: Flip probability [0., 1.]

    Returns:
        images: Image [B, H, W, C] or [H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
    """
    labels = labels.copy()

    boxes = labels.get("bboxes", None)
    masks = labels.get("masks", None)

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

    if not masks == None:
        labels["masks"] = tf.reverse(masks, axis = [start_axis+1])

    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=seed) >= prob:
        return images, labels

    im_width = tf.cast(im_width, boxes.dtype)
    im_height = tf.cast(im_height, boxes.dtype)

    xmin, ymin, xmax, ymax = tf.split(boxes, 4, axis=-1)

    xmin = xmin / im_width
    ymin = ymin / im_height
    xmax = xmax / im_width
    ymax = ymax / im_height

    images = tf.image.flip_left_right(images)
    new_xmin = 1 - xmax
    new_xmax = 1 - xmin
    new_ymin = ymin
    new_ymax = ymax

    xmin = new_xmin * im_width
    ymin = new_ymin * im_height
    xmax = new_xmax * im_width
    ymax = new_ymax * im_height

    labels["bboxes"] = tf.concat([xmin, ymin, xmax, ymax], axis=-1)
    return images, labels


@register_transforms
@tf.function
def mixup(images: Tensor, labels: Dict[str,Tensor],  generator: tf.random.Generator, seed: int, prob: float=0, alpha: float=0.2, beta: float=0.2) -> Tuple[Tensor, Dict[str,Tensor]]:
    """
    Mixup a pair of random images from a batch of data

    Args:
        images: Image [B, H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
        alpha: alpha transparent range
        beta: beta transparent range
        generator: TF random generator

    Returns:
        images: Image [B, H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
    """

    labels = labels.copy()

    boxes = labels['bboxes']
    cls = labels['classes']
    
    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=seed) >= prob:
        return images, labels

    batch_size = tf.shape(images)[0]

    image_pairs = generator.uniform(
        (batch_size, 1),
        minval=0,
        maxval=batch_size,
        dtype=tf.int32
    )
    image_pairs = tf.squeeze(image_pairs, axis=1)

    # augment images
    sample_alpha = tf.random.gamma(
        (batch_size,),
        alpha=alpha,
    )
    sample_beta = tf.random.gamma(
        (batch_size,),
        alpha=beta,
    )

    lambda_ = sample_alpha / (sample_alpha + sample_beta)
    lambda_ = tf.reshape(lambda_, [-1, 1, 1, 1])

    mixup_images = tf.gather(images, image_pairs)
    output_images = lambda_ * images + (1 - lambda_) * mixup_images

    # augment boxes
    mixup_boxes = tf.gather(boxes, image_pairs)
    output_boxes = tf.concat([boxes, mixup_boxes], axis=1)

    # augment labels
    mixup_cls = tf.gather(cls, image_pairs)
    output_cls = tf.concat([cls, mixup_cls], axis=1)

    labels['bboxes'] = output_boxes
    labels['classes'] = output_cls

    return output_images, labels


@register_transforms
@tf.function
def mosaic(images: Tensor, labels: Dict[str,Tensor], generator: tf.random.Generator, seed: int, prob: float=0) -> Tuple[Tensor, Dict[str,Tensor]]:
    """
    Tile 4 random images from a batch of data

    Args:
        images: Image [B, H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
        default_pad_value: Constant value to pad the boxes and classes
        generator: TF random generator

    Returns:
        images: Image [B, H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
    """

    labels = labels.copy()

    boxes = labels.get('bboxes', None)
    cls = labels.get('classes', None)
    masks = labels.get('masks', None)
    
    # augment masks
    if not masks == None:

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

        masks = tf.slice(output_masks, \
                             [0, int(top/mask_ratio), int(left/mask_ratio), 0], \
                             [batch_size, mask_height, mask_width, mask_count*4])
        labels["masks"] = tf.image.convert_image_dtype(masks, tf.float32)
    
    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=seed) >= prob:
        return images, labels

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

    # augment boxes and labels
    batch_size = tf.shape(boxes)[0]
    box_count = tf.shape(boxes)[1]

    imgsz = tf.cast(imgsz, boxes.dtype)
    left = tf.cast(left, boxes.dtype)
    top = tf.cast(top, boxes.dtype)

    mosaic_boxes = tf.gather(boxes, image_pairs)
    mosaic_boxes = tf.split(mosaic_boxes, 4, axis=1)

    mosaic_cls = tf.gather(cls, image_pairs)
    mosaic_cls = tf.split(mosaic_cls, 4, axis=1)

    cls_list = []
    for m_cls in mosaic_cls:
        m_cls = tf.squeeze(m_cls, axis=1)
        cls_list.append(m_cls)
    cls = tf.concat(cls_list, axis=1)

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

    labels["bboxes"] = boxes
    labels["classes"] = cls

    return output_images, labels


@register_transforms
@tf.function
def random_perspective(images: Tensor, labels: Dict[str, Tensor], perspective: float=0.005, degree: float=0., scale: float=0.5, shear: float=0.0, translate: float=0.1, fill_value: int=0):
    """
    Random perspective transform on a batch of images

    Args:
        images: Image [B, H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
        perspective: Max perspective value range
        degree: Max rotation degree
        scale: Max scale value
        shear: Max shear value
        translate: Max translation value
        fill_value: Contant value to fill the empty spaces
        default_pad_value: Constant value to pad the boxes and classes

    Returns:
        images: Image [B, H, W, C]
        labels: Dict of "cls", "bboxes" and "masks"
    """

    labels = labels.copy()

    h = tf.shape(images)[1]
    w = tf.shape(images)[2]

    boxes = labels.get('bboxes', None)
    cls = labels.get('classes', None)
    masks = labels.get('masks', None)

    batch_size = tf.shape(boxes)[0]

    # Center
    C = tf.convert_to_tensor([
        [1., 0., -w / 2],
        [0., 1., -h / 2],
        [0., 0., 1.]
    ], dtype=tf.float32)

    # Perspective
    p_rand = tf.random.uniform((2,), -perspective, perspective)
    P = tf.eye(3, dtype=tf.float32)
    P = tf.tensor_scatter_nd_update(P, [[2,0], [2,1]], p_rand)

    # Rotation
    r_rand = tf.random.uniform([], -degree, degree)
    R = _get_rotation_matrix(r_rand, h, w)

    # Scale
    x_rand = tf.random.uniform([], -scale, scale)
    X = tf.eye(3, dtype=tf.float32)
    X = tf.tensor_scatter_nd_update(X, [[0,0], [1,1]], [x_rand, x_rand])

    # Shear
    s_rand = tf.math.tan(tf.random.uniform((2,), -shear, shear) * pi / 180)
    S = tf.eye(3, dtype=tf.float32)
    S = tf.tensor_scatter_nd_update(S, [[0,1], [1,0]], s_rand)

    # Translation
    t_rand = tf.random.uniform((2,), 0.5 - translate, 0.5 + translate)

    image_t_rand = t_rand * [w, h]
    T = tf.eye(3, dtype=tf.float32)
    T = tf.tensor_scatter_nd_update(T, [[0,2], [1,2]], image_t_rand)

    # Image Transform Matrix
    # M = T @ S @ P @ C @ R
    M = T @ S @ C @ R
    image_M = tf.reshape(M, [1, -1])[:, :-1]

    images = tf.raw_ops.ImageProjectiveTransformV3(
                images=images,
                output_shape=tf.shape(images)[1:3],
                fill_value=tf.convert_to_tensor(fill_value, tf.float32),
                transforms=image_M,
                interpolation="BILINEAR",
                fill_mode="CONSTANT",
            )

    # Augment masks
    if not masks == None:
        mask_h = tf.shape(masks)[1]
        mask_w = tf.shape(masks)[2]

        mask_C = tf.convert_to_tensor([
            [1., 0., -mask_w / 2],
            [0., 1., -mask_h / 2],
            [0., 0., 1.]
        ], dtype=tf.float32)
        mask_R = _get_rotation_matrix(r_rand, mask_h, mask_w)

        mask_t_rand = t_rand * [mask_w, mask_h]
        mask_T = tf.eye(3, dtype=tf.float32)
        mask_T = tf.tensor_scatter_nd_update(mask_T, [[0,2], [1,2]], mask_t_rand)

        # Mask Transform Matrix
        mask_M = mask_T @ S @ mask_C @ mask_R
        mask_M = tf.reshape(mask_M, [1, -1])[:, :-1]

        masks = masks / 255.
        masks = tf.raw_ops.ImageProjectiveTransformV3(
                    images=masks,
                    output_shape=tf.shape(masks)[1:3],
                    fill_value=tf.convert_to_tensor(fill_value, tf.float32),
                    transforms=mask_M,
                    interpolation="BILINEAR",
                    fill_mode="CONSTANT",
                )
        masks = tf.clip_by_value(masks, 0., 1.) * 255.

    # augment boxes
    n = tf.shape(boxes)[1]

    gathered_boxes = tf.gather(boxes, [0, 1, 2, 3, 0, 3, 2, 1], axis=-1)

    xy = tf.concat([
            tf.reshape(gathered_boxes, (batch_size, n*4, 2)),
            tf.ones((batch_size, n*4, 1), dtype=boxes.dtype)
        ], axis=-1)

    M = tf.linalg.inv(M)
    xy = xy @ tf.transpose(M)
    xy = tf.reshape(xy[:, :, :2] / xy[:, :, 2:3], (batch_size, n, 8))

    x = tf.gather(xy, [0, 2, 4, 6], axis=-1)
    y = tf.gather(xy, [1, 3, 5, 7], axis=-1)

    boxes = tf.transpose(tf.reshape(tf.concat([
                tf.reduce_min(x, axis=2),
                tf.reduce_min(y, axis=2),
                tf.reduce_max(x, axis=2),
                tf.reduce_max(y, axis=2)
            ], axis=-1), (batch_size, 4, n)), (0, 2, 1))

    labels["bboxes"] = boxes
    labels["classes"] = cls

    if not masks == None:
        labels["masks"] = masks

    return images, labels


