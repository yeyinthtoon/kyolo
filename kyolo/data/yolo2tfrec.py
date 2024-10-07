from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from kyolo.utils.bounding_box_utils import xywh2xyxy
from kyolo.utils.data_utils import (tf_bytes_feature, check_corrupt_jpg,
                                    tf_float_feature, tf_int64_feature)
from PIL import Image, ImageDraw, ImageOps


def get_label_features(labels, label_file, img_size, save_masks=True, save_polys=False):
    """
    Read yolo label files and convert them to tf format
    """
    classes = []
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    masks = []
    polygons = []

    labels = labels.split("\n")
    _, indices = np.unique(labels, return_index=True)
    ind = 0
    for ind, label in enumerate(labels):
        if not label:
            continue
        if ind not in indices:
            continue
        splitted_label = label.split()
        cls = int(float(splitted_label[0]))
        classes.append(cls)

        if len(splitted_label) > 5:
            polygon = np.array(splitted_label[1:], dtype="float32")
            if save_polys:
                polygons.append(polygon)

            polygon = polygon.reshape(-1, 2)
            if save_masks:
                mask = Image.new(mode="1", size=img_size)
                fillpoly_pts = np.int32(polygon * img_size)
                draw = ImageDraw.Draw(mask)
                draw.polygon(fillpoly_pts.flatten().tolist(), fill=1, outline=1)
                mask_bytes = BytesIO()
                mask.save(mask_bytes, format="PNG", bits=1, optimize=True)
                masks.append(mask_bytes.getvalue())

            xmins.append(np.min(polygon[:, 0], axis=-1))
            xmaxs.append(np.max(polygon[:, 0], axis=-1))
            ymins.append(np.min(polygon[:, 1], axis=-1))
            ymaxs.append(np.max(polygon[:, 1], axis=-1))

        else:
            bbox = np.array(splitted_label[1:], dtype="float32")
            xmin, ymin, xmax, ymax = xywh2xyxy(bbox)
            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)

    if ind + 1 > len(indices):
        print(f"Duplicate annotation in label file {label_file} removed!")

    return classes, xmins, xmaxs, ymins, ymaxs, masks, polygons


def yolo2tfrec(
    img_files,
    images_per_record,
    label_map,
    output_path_prefix,
    save_polys=False,
    save_masks=True,
) -> None:
    """
    Converts yolo data to tfrecords
    """
    corrupt_images = 0
    tiny_images = 0
    no_label = 0
    empty_label = 0
    if not output_path_prefix.parent.exists():
        output_path_prefix.parent.mkdir()

    for count, start_i in enumerate(range(0, len(img_files), images_per_record)):
        with tf.io.TFRecordWriter(f"{output_path_prefix}_{count}.tfrecord") as writer:
            for img_file in img_files[start_i : start_i + images_per_record]:
                if check_corrupt_jpg(img_file):
                    corrupt_images += 1
                    continue
                img = Image.open(img_file)
                width, height = img.size
                if width < 10 and height < 10:
                    print(f"Skipped tiny image: {img_file}")
                    tiny_images += 1
                    continue

                img = ImageOps.exif_transpose(img)
                img_format = img_file.suffix[1:]
                img_byte = BytesIO()
                if img_format.lower() == "jpg":
                    img.save(img_byte, format="JPEG")
                else:
                    img.save(img_byte, format=img_format.upper())

                label_file = Path(img_file.as_posix().replace("/images/", "/labels/"))
                label_file = label_file.with_suffix(".txt")
                if not label_file.is_file():
                    print(f"No label file found for {img_file}.")
                    no_label += 1
                    continue

                labels = label_file.read_text().strip()
                if not labels:
                    print(f"Empty label file {label_file}.")
                    empty_label += 1
                    continue

                classes, xmins, xmaxs, ymins, ymaxs, masks, polygons = (
                    get_label_features(
                        labels,
                        label_file,
                        img.size,
                        save_masks=save_masks,
                        save_polys=save_polys,
                    )
                )

                if classes:
                    if max(classes) > len(label_map) - 1:
                        raise ValueError(
                            f"Class {max(classes)} is greater than maximum possible class {len(label_map)-1}."
                        )

                class_texts = [label_map[cls].encode("utf8") for cls in classes]

                features = {
                    "image/height": tf_int64_feature(height),
                    "image/width": tf_int64_feature(width),
                    "image/filename": tf_bytes_feature(img_file.name.encode("utf8")),
                    "image/encoded": tf_bytes_feature(img_byte.getvalue()),
                    "image/format": tf_bytes_feature(img_format.encode("utf8")),
                    "image/object/bbox/xmin": tf_float_feature(xmins, True),
                    "image/object/bbox/xmax": tf_float_feature(xmaxs, True),
                    "image/object/bbox/ymin": tf_float_feature(ymins, True),
                    "image/object/bbox/ymax": tf_float_feature(ymaxs, True),
                    "image/object/class/text": tf_bytes_feature(class_texts, True),
                    "image/object/class/label": tf_int64_feature(classes, True),
                }
                if save_masks:
                    features.update(
                        {"image/object/mask/binary": tf_bytes_feature(masks, True)}
                    )
                if save_polys:
                    features.update(
                        {"image/object/mask/polygon": tf_float_feature(polygons, True)}
                    )

                tf_example = tf.train.Example(
                    features=tf.train.Features(feature=features)
                )
                writer.write(tf_example.SerializeToString())
