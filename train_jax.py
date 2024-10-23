from pathlib import Path

import hydra
import jax
import keras
import numpy as np
from keras import callbacks
from kyolo.data.dataset import build_tfrec_dataset
from kyolo.model.build import build_model
from kyolo.model.losses import BCELoss, BoxLoss, DFLLoss
from kyolo.utils.callbacks import PyCOCOCallback
from kyolo.utils.optimizers import MultiLRAdam
from omegaconf import OmegaConf


@hydra.main(config_path="kyolo/configs", config_name="config", version_base=None)
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
    train_tfrec_path = Path(config.dataset.train_tfrecs)
    train_tfrecs = list(map(str, train_tfrec_path.glob("*.tfrecord")))
    train_dataset = build_tfrec_dataset(
        np.asarray(train_tfrecs), data_config, config.task, "train"
    )
    val_tfrec_path = Path(config.dataset.val_tfrecs)
    val_tfrecs = list(map(str, val_tfrec_path.glob("*.tfrecord")))
    val_dataset = build_tfrec_dataset(
        np.asarray(val_tfrecs), data_config, config.task, "val"
    )

    model = build_model(OmegaConf.to_object(config), True)

    decay = keras.optimizers.schedules.CosineDecay(
        0.0,
        15 * 97,
        alpha=0.0,
        name="CosineDecay",
        warmup_target=1e-3,
        warmup_steps=15 * 3,
    )
    bias_decay = keras.optimizers.schedules.CosineDecay(
        0.01,
        15 * 97,
        alpha=0.0,
        name="CosineDecay",
        warmup_target=1e-3,
        warmup_steps=15 * 3,
    )
    learning_rates = {
        "bias": bias_decay,
    }
    optimizer = MultiLRAdam(
        learning_rate=decay, learning_rates=learning_rates, weight_decay=0.0005
    )
    optimizer.exclude_from_weight_decay(
        var_names=["gamma", "beta", "moving_mean", "moving_variance", "bias"]
    )
    model.compile(
        box_loss=BoxLoss,
        classification_loss=BCELoss,
        dfl_loss=DFLLoss,
        box_loss_weight=7.5,
        classification_loss_weight=0.5,
        dfl_loss_weight=1.5,
        optimizer=optimizer,
        jit_compile=True,
        steps_per_execution=2,
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
