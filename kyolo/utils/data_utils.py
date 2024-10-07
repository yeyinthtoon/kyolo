from pathlib import Path
from typing import List, Union

import tensorflow as tf


def check_corrupt_jpg(img_file: Path) -> bool:
    """
    Check whether if a file is jpg and if it is check whether if it's corrupt
    """
    if img_file.suffix.lower() in [".jpg", ".jpeg"]:
        if not img_file.read_bytes().endswith(b"\xff\xd9"):
            print(f"Skipped corrupt jpeg: {img_file}")
            return True
    return False


def tf_bytes_feature(value: Union[bytes, List[bytes]], list_input: bool = False):
    """
    Returns a bytes_list from a string / byte.

    Args:
        value: bytes or list of bytes
        list_input: bool, whether the input is list of bytes or single feature

    Returns:
        tensorflow bytes list feature
    """
    if not list_input:
        if isinstance(value, type(tf.constant(0))):
            value = (
                value.numpy()
            )  # BytesList won't unpack a string from an EagerTensor.
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def tf_float_feature(value: Union[float, List[float]], list_input=False):
    """
    Returns a float_list from a float / double.

    Args:
        value: bytes or list of bytes
        list_input: bool, whether the input is list of float or single float

    Returns:
        tensorflow float list feature
    """
    if not list_input:
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def tf_int64_feature(value: Union[int, List[int]], list_input=False):
    """
    Returns an int64_list from a bool / enum / int / uint.

    Args:
        value: int or list of int
        list_input: bool, whether the input is list of int or single int

    Returns:
        tensorflow int64 list feature
    """
    if not list_input:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
