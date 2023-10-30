from lib.data.s3 import load_data
from tqdm import tqdm
from lib.training.utils import (
    train_epoch,
    validate_after_epoch,
    make_val_test_predictions,
    make_test_predictions,
)
from lib.utils import seed_everything, load_config
from lib.data.dataset import cross_val_split, make_dataloader
from lib.training.utils import init_nn_stuff


def main(config_path):
    cfg = load_config(config_path)
    cfg_name = cfg["name"]

    seed_everything(cfg["seed"])
    tag_data, track_idx2embeds = load_data(cfg)

    test_dataloader = make_dataloader(
        tag_data["test"],
        track_idx2embeds,
        cfg,
        testing=True,
    )
    if cfg.get("use_cv", True):
        cv = cross_val_split(tag_data["train"], track_idx2embeds, cfg)
        cv_min_score_to_save_predictions = cfg.get("cv_min_score_to_save_predictions", 0.0)
        epochs = cfg.get("cv_n_epochs", 15)
        for fold_idx, (train_dataloader, val_dataloader) in enumerate(cv):
            model, criterion, optimizer, scheduler = init_nn_stuff(cfg)
            has_predict = False
            for epoch in tqdm(range(epochs)):
                train_epoch(model, train_dataloader, criterion, optimizer, scheduler)
                score = validate_after_epoch(model, val_dataloader)
                if score > cv_min_score_to_save_predictions:
                    make_val_test_predictions(
                        model,
                        val_dataloader,
                        test_dataloader,
                        cfg_name,
                        fold_idx,
                        epoch,
                        score,
                    )
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
                )
    if cfg.get("use_solo", True):
        train_dataloader = make_dataloader(
            tag_data["train"],
            track_idx2embeds,
            cfg,
            testing=False,
        )
        model, criterion, optimizer, scheduler = init_nn_stuff(cfg)
        epochs = cfg.get("solo_n_epochs", 15)
        for epoch in tqdm(epochs):
            train_epoch(model, train_dataloader, criterion, optimizer, scheduler)
            if epochs - epoch <= cfg.get("solo_save_last_n_epochs", 1):
                make_test_predictions(
                    model,
                    test_dataloader,
                    path="predictions_test",
                    suffix=f"cfg={cfg_name}__fold_idx=-1__epoch={epoch}__score=-1",
                )


if __name__ == "__main__":
    main("configs/3.yaml")
