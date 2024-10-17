import glob
from pathlib import Path

import hydra
import jax
import keras
import numpy as np
from keras import callbacks
from kyolo.data.dataset import build_tfrec_dataset
from kyolo.model.build import build_model
from kyolo.model.losses import bce_yolo, box_loss_yolo_jit, dfl_loss_yolo_jit
from kyolo.utils.callbacks import PyCOCOCallback
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config):
    devices = jax.devices()
    accelerator_type = jax.default_backend()
    print("Devices:", devices)
    distribution = keras.distribution.DataParallel(devices=devices)
    keras.distribution.set_distribution(distribution)
    outdir = Path(config.save_dir)

    if config.common.mixed_precision:
        if accelerator_type == "tpu":
            data_type = "mixed_bfloat16"
        else:
            data_type = "mixed_float16"
        keras.mixed_precision.set_global_policy(data_type)

    data_config = OmegaConf.to_object(config.data)
    train_tfrec_path = str(Path(config.dataset.train_tfrecs))
    train_tfrecs = glob.glob(f"{train_tfrec_path}*.tfrecord")
    train_dataset = build_tfrec_dataset(
        np.asarray(train_tfrecs), data_config, config.common.task, "train"
    )
    val_tfrec_path = str(Path(config.dataset.val_tfrecs))
    val_tfrecs = glob.glob(f"{val_tfrec_path}*.tfrecord")
    val_dataset = build_tfrec_dataset(
        np.asarray(val_tfrecs), data_config, config.common.task, "val"
    )

    model = build_model(OmegaConf.to_object(config), True)

    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.004,
    )
    optimizer.exclude_from_weight_decay(var_names=["bn", "bias"])
    model.compile(
        box_loss=box_loss_yolo_jit,
        classification_loss=bce_yolo,
        dfl_loss=dfl_loss_yolo_jit,
        box_loss_weight=7.5,
        classification_loss_weight=0.5,
        dfl_loss_weight=1.5,
        optimizer=optimizer,
        jit_compile=True,
        steps_per_execution=8,
    )
    best_ckpt_callback = callbacks.ModelCheckpoint(
        filepath=outdir / "best.keras",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
    )

    last_ckpt_callback = callbacks.ModelCheckpoint(
        filepath=outdir / "last.keras",
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
    )

    early_stop_callback = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
    )

    tensorboard_callback = callbacks.TensorBoard(
        log_dir=outdir / "logs",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    )
    val_callback = PyCOCOCallback(
        validation_data=val_dataset,
        bounding_box_format="xyxy",
        pred_key="main",
        nms_conf=0.5,
        nms_iou=0.3,
        max_detection=100,
        cache=False,
    )
    model.fit(
        train_dataset,
        epochs=config.common.max_epoch,
        validation_data=val_dataset,
        callbacks=[
            val_callback,
            last_ckpt_callback,
            best_ckpt_callback,
            tensorboard_callback,
            early_stop_callback,
            callbacks.TerminateOnNaN(),
        ],
    )

if __name__ == "__main__":
    main()