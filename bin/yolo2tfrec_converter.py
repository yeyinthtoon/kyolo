import argparse
from pathlib import Path

import yaml
from kyolo.data.yolo2tfrec import yolo2tfrec


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_dir", help="Path to torch weights")
    parser.add_argument(
        "--save_polys", default=False, help="Whether to save polygons if there's any"
    )
    parser.add_argument(
        "--save_masks", default=True, help="Whether to save masks if there's any"
    )
    parser.add_argument("--images_per_record", type=int, help="Images per tfrec")
    parser.add_argument(
        "--tfrec_dir", help="Path and prefix of tfrec e.g. /data/coco"
    )

    args = parser.parse_args()
    return args


def main(args):
    yolo_dir = Path(args.yolo_dir)
    exts = [
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".dng",
        ".mpo",
        ".tif",
        ".tiff",
        ".webp",
        ".pfm",
    ]

    with open(yolo_dir / "data.yaml", "r", encoding="utf-8") as f:
        data_dict = yaml.safe_load(f)

    label_map = data_dict["names"]

    save_polys = args.save_polys
    save_masks = args.save_masks
    images_per_record = args.images_per_record
    tfrec_dir = Path(args.tfrec_dir)
    dataset_yaml = {
        "num_class": len(label_map)
    }

    for mode in ["train", "test", "val"]:
        img_files = [
            img_file
            for img_file in yolo_dir.glob("**/images/**/*.*")
            if img_file.suffix.lower() in exts and f"/{mode}/" in img_file.as_posix()
        ]
        if len(img_files) < 1:
            print(f"No {mode} files")
            continue
        out_dir = tfrec_dir / mode
        dataset_yaml.update({
            f"{mode}_tfrecs" : str(out_dir)
        })
        yolo2tfrec(
            img_files, images_per_record, label_map, out_dir, save_polys, save_masks
        )

    dataset_yaml.update({
        "class_map": label_map
    })
    with open(tfrec_dir/"dataset.yaml","w",encoding="utf-8") as f:
        yaml.dump({"dataset": dataset_yaml}, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    converter_args = parse_arguments()
    main(converter_args)
