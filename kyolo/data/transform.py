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