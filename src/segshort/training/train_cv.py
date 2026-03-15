import argparse
import glob
import os

import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

from segshort.training.baseline_losses import (
    DiceCELoss,
    GroupDROLoss,
)
from segshort.training.datasets import WaterbirdsDataset, CocoCatsDogsDataset


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(model_name, num_classes, device):
    encoder = "resnet50" if model_name == "resnet" else "mit_b2"
    model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=num_classes)
    return model.to(device)


def evaluate(model, loader, device, num_classes):
    model.eval()
    inter = torch.zeros(num_classes, dtype=torch.int64)
    union = torch.zeros(num_classes, dtype=torch.int64)
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            for c in range(num_classes):
                pred_c = preds == c
                target_c = masks == c
                inter[c] += (pred_c & target_c).sum().item()
                union[c] += (pred_c | target_c).sum().item()
    iou = inter / (union + 1e-8)
    return iou.mean().item()


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )


def load_waterbirds(
    data_root,
    bg_aug_kind="none",
):
    meta_path = os.path.join(data_root, "images", "metadata.csv")
    df = pd.read_csv(meta_path)
    df["is_water"] = df["place_filename"].str.contains("ocean|lake|river|sea", case=False)
    df["spurious_attr"] = df["is_water"].astype(int)
    df["group_id"] = df["y"] * 2 + df["spurious_attr"]
    assert (df["group_id"] == df["y"] * 2 + df["place"]).all()
    train_df = df[df["split"] == 0]
    val_df = df[df["split"] == 1]
    test_df = df[df["split"] == 2]
    return {
        "train": WaterbirdsDataset(
            data_root,
            train_df["img_filename"].tolist(),
            train_df["group_id"].tolist(),
            num_classes=3,
            augment=True,
            bg_aug_kind=bg_aug_kind,
        ),
        "val": WaterbirdsDataset(
            data_root,
            val_df["img_filename"].tolist(),
            val_df["group_id"].tolist(),
            num_classes=3,
            augment=False,
        ),
        "test": WaterbirdsDataset(
            data_root,
            test_df["img_filename"].tolist(),
            test_df["group_id"].tolist(),
            num_classes=3,
            augment=False,
        ),
        "num_classes": 3,
    }


def load_coco_catsdogs(
    coco_root,
    annotations_train,
    annotations_val,
    annotations_test,
    resize_size,
    bg_aug_kind="none",
):
    train = CocoCatsDogsDataset(
        annotations_train,
        coco_root,
        resize_size=resize_size,
        augment=True,
        bg_aug_kind=bg_aug_kind,
    )
    val = CocoCatsDogsDataset(
        annotations_val,
        coco_root,
        resize_size=resize_size,
        augment=False,
    )
    test = CocoCatsDogsDataset(
        annotations_test,
        coco_root,
        resize_size=resize_size,
        augment=False,
    )
    return {"train": train, "val": val, "test": test, "num_classes": 3}


def resolve_coco_annotation_paths(data_root, train_path, val_path, test_path):
    if data_root is not None:
        annotations_dir = os.path.join(data_root, "annotations")
    else:
        annotations_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "annotations", "coco_cd")
        )
    resolved_train = train_path
    if resolved_train is None:
        matches = sorted(glob.glob(os.path.join(annotations_dir, "instances_train_corr*.json")))
        if len(matches) == 1:
            resolved_train = matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f"Multiple train annotation files found under {annotations_dir}; "
                "pass --coco-annotations-train explicitly."
            )
    resolved_val = val_path or os.path.join(annotations_dir, "instances_val_balanced.json")
    resolved_test = test_path or os.path.join(annotations_dir, "instances_test_balanced.json")
    return resolved_train, resolved_val, resolved_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default=None,
        help=(
            "Dataset root. For coco_cd, expects annotations/instances_*.json under this "
            "directory; if omitted, falls back to repo-local annotations/coco_cd/."
        ),
    )
    parser.add_argument("--dataset", required=True, choices=["waterbirds_seg", "coco_cd"])
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--coco-root", default=None, help="COCO root containing train2017/val2017.")
    parser.add_argument("--coco-annotations-train", default=None, help="COCO annotations JSON for training.")
    parser.add_argument("--coco-annotations-val", default=None, help="COCO annotations JSON for validation.")
    parser.add_argument("--coco-annotations-test", default=None, help="COCO annotations JSON for testing.")
    parser.add_argument("--model-name", default="resnet")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--kind",
        default="CE",
        choices=["CE", "DICECE", "GROUPDRO", "CUTMIX"],
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(enabled=device.type == "cuda")

    bg_aug_kind = "cutmix" if args.kind.upper() == "CUTMIX" else "none"

    if args.dataset == "waterbirds_seg":
        if args.data_root is None:
            raise ValueError("--data_root is required for waterbirds_seg.")
        data = load_waterbirds(
            args.data_root,
            bg_aug_kind=bg_aug_kind,
        )
    else:
        if args.coco_root is None:
            raise ValueError("--coco-root is required for coco_cd.")
        (
            args.coco_annotations_train,
            args.coco_annotations_val,
            args.coco_annotations_test,
        ) = resolve_coco_annotation_paths(
            args.data_root,
            args.coco_annotations_train,
            args.coco_annotations_val,
            args.coco_annotations_test,
        )
        if args.coco_annotations_train is None or args.coco_annotations_val is None:
            raise ValueError(
                "Provide shared COCO-CD annotations via repo-local annotations/coco_cd/, "
                "--data_root, or explicit --coco-annotations-train/--coco-annotations-val."
            )
        data = load_coco_catsdogs(
            args.coco_root,
            args.coco_annotations_train,
            args.coco_annotations_val,
            args.coco_annotations_test,
            args.resize_size,
            bg_aug_kind=bg_aug_kind,
        )

    train_loader = make_loader(data["train"], args.batch_size, shuffle=True)
    val_loader = make_loader(data["val"], args.batch_size, shuffle=False)
    test_loader = make_loader(data["test"], args.batch_size, shuffle=False)

    model = make_model(args.model_name, data["num_classes"], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    kind = args.kind.upper()
    if kind in ["CE", "CUTMIX"]:
        loss_name, criterion = "ce", torch.nn.CrossEntropyLoss()
    elif kind == "DICECE":
        loss_name, criterion = "dicece", DiceCELoss(include_background=False)
    else:
        loss_name, criterion = "groupdro", GroupDROLoss(n_groups=4)
    criterion = criterion.to(device)

    best_val = float("inf")
    epochs_no_improve = 0
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        model.train()
        running_loss = 0.0
        for imgs, masks, group_ids in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)
            group_ids = group_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(imgs)
                if loss_name == "groupdro":
                    loss, _ = criterion(logits, masks, group_ids)
                else:
                    loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        val_loss = 0.0
        with torch.no_grad():
            model.eval()
            for imgs, masks, group_ids in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                group_ids = group_ids.to(device)
                with autocast(device_type="cuda", enabled=device.type == "cuda"):
                    logits = model(imgs)
                    if loss_name == "groupdro":
                        loss, _ = criterion(logits, masks, group_ids)
                    else:
                        loss = criterion(logits, masks)
                val_loss += loss.item()
        val_loss = val_loss / max(1, len(val_loader))
        val_miou = evaluate(model, val_loader, device, data["num_classes"])
        print(
            f"Epoch {epoch}: loss={running_loss / len(train_loader):.4f} "
            f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                ckpt_name = f"best_model_{args.model_name}_{args.kind.lower()}.pt"
                torch.save(model.state_dict(), os.path.join(args.output_dir, ckpt_name))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(args.output_dir, f"last_model_{args.model_name}_{args.kind.lower()}.pt"),
        )

    test_miou = evaluate(model, test_loader, device, data["num_classes"])
    print(f"Test mIoU={test_miou:.4f}")


if __name__ == "__main__":
    main()
