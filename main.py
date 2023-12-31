from lib.data.s3 import load_data
from tqdm import tqdm
import click
import torchinfo
from lib.training.utils import (
    train_epoch,
    validate_after_epoch,
    make_val_test_predictions,
    make_test_predictions,
)
from lib.utils import seed_everything, load_config
from lib.data.dataset import cross_val_split, make_dataloader
from lib.training.utils import init_nn_stuff


def run_cfg(config_path, files_mode=True):
    cfg = load_config(config_path)
    cfg_name = cfg["name"]

    seed_everything(cfg["seed"])
    tag_data, track_idx2embeds, track_idx2knn = load_data(cfg)

    test_dataloader = make_dataloader(
        tag_data["test"],
        track_idx2embeds,
        track_idx2knn,
        cfg,
        testing_dataset=True,
        testing_collator=not cfg.get("test_augmentations", False),
    )
    model_info_was_printed = False
    if cfg.get("use_cv", True):
        cv = cross_val_split(tag_data["train"], track_idx2embeds, track_idx2knn, cfg)
        epochs = cfg.get("cv_n_epochs", 15)
        for fold_idx, (train_dataloader, val_dataloader) in enumerate(cv):
            model, criterion, optimizer, scheduler, ema = init_nn_stuff(cfg)

            if not model_info_was_printed:
                print(torchinfo.summary(model))
                model_info_was_printed = True

            best_score = cfg.get("cv_min_score_to_save_predictions", 0.0)
            has_predict = False
            for epoch in tqdm(range(epochs)):
                train_epoch(
                    model, train_dataloader, criterion, optimizer, scheduler, ema=ema
                )
                score = validate_after_epoch(model, val_dataloader, ema=ema)
                if not files_mode:
                    continue
                if score > best_score:
                    make_val_test_predictions(
                        model,
                        val_dataloader,
                        test_dataloader,
                        cfg_name,
                        fold_idx,
                        epoch,
                        score,
                        ema=ema,
                    )
                    best_score = score
                    has_predict = True
            if not has_predict:
                make_val_test_predictions(
                    model,
                    val_dataloader,
                    test_dataloader,
                    cfg_name,
                    fold_idx,
                    epoch,
                    score,
                    ema=ema,
                )
    if cfg.get("use_solo", True) and files_mode:
        train_dataloader = make_dataloader(
            tag_data["train"],
            track_idx2embeds,
            track_idx2knn,
            cfg,
            testing_dataset=False,
            testing_collator=False,
        )
        model, criterion, optimizer, scheduler, ema = init_nn_stuff(cfg)
        if not model_info_was_printed:
            print(torchinfo.summary(model))
        epochs = cfg.get("solo_n_epochs", 15)
        for epoch in tqdm(range(epochs)):
            train_epoch(
                model, train_dataloader, criterion, optimizer, scheduler, ema=ema
            )
            if epochs - epoch <= cfg.get("solo_save_last_n_epochs", 1):
                test_preds = make_test_predictions(
                    model,
                    test_dataloader,
                    path="predictions_test",
                    suffix=f"cfg={cfg_name}__fold_idx=-1__epoch={epoch}__score=-1",
                    ema=ema,
                )
        if cfg.get("distillation", False):
            dataloader = make_dataloader(
                test_preds.rename(columns={"prediction": "tags"}),
                track_idx2embeds,
                track_idx2knn,
                cfg,
                testing_dataset=False,
                testing_collator=False,
            )
            for epoch in tqdm(range(2)):
                for g in optimizer.param_groups:
                    g["lr"] = 0.000001
                train_epoch(
                    model, dataloader, criterion, optimizer, scheduler=None, ema=ema
                )
            make_test_predictions(
                model,
                test_dataloader,
                path="predictions_test",
                suffix=f"cfg={cfg_name}__fold_idx=-2__epoch={epoch}__score=-2",
                ema=ema,
            )


def main(config_paths, files_mode=True):
    for config_path in config_paths:
        run_cfg(config_path, files_mode)


@click.command()
@click.option("--cfg_path", "-cfg", help="config path", required=True)
def training_demo(cfg_path):
    main([cfg_path])


if __name__ == "__main__":
    training_demo()
