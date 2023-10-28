from lib.data.s3 import load_data
from tqdm import tqdm
from lib.const import DEVICE
from lib.training.utils import train_epoch, validate_after_epoch, make_test_predictions
from lib.utils import seed_everything, load_config, make_instance
from lib.training.optimizer import get_grouped_parameters
from lib.data.dataset import cross_val_split, make_dataloader


def main(config_path):
    cfg = load_config(config_path)
    cfg_name = cfg["name"]

    seed_everything(cfg["seed"])

    model = make_instance(cfg["model"], **cfg["model_params"])
    criterion = make_instance(cfg["criterion"])

    epochs = cfg["n_epochs"]
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    optimizer = make_instance(
        cfg["optimizer"],
        get_grouped_parameters(model, cfg["lr"], cfg["lr_alpha"]),
        **cfg["optimizer_params"],
    )

    if "scheduler" in cfg:
        scheduler = make_instance(
            cfg["scheduler"], optimizer, **cfg["scheduler_params"]
        )
    else:
        scheduler = None

    tag_data, track_idx2embeds = load_data(cfg)

    cv = cross_val_split(tag_data["train"], track_idx2embeds, cfg)
    test_dataloader = make_dataloader(
        tag_data["test"],
        track_idx2embeds,
        cfg,
        dataset_testing=True,
        collator_testing=True,
    )
    min_val_score = cfg["best_score"]
    has_predict = False
    for fold_idx, (train_dataloader, val_dataloader) in enumerate(cv):
        for epoch in tqdm(range(epochs)):
            train_epoch(model, train_dataloader, criterion, optimizer, scheduler)
            score = validate_after_epoch(model, val_dataloader)
            if score > min_val_score:
                make_test_predictions(
                    model,
                    test_dataloader,
                    path="predictions",
                    suffix=f"cfg={cfg_name}__fold_idx={fold_idx}__epoch={epoch}__score={score:.5f}",
                )
                has_predict = True
        if not has_predict:
            make_test_predictions(
                model,
                test_dataloader,
                path="predictions",
                suffix=f"cfg={cfg_name}__fold_idx={fold_idx}__epoch={epoch}__score={score:.5f}",
            )


if __name__ == "__main__":
    main("configs/3.yaml")
