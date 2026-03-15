import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from segshort.training.train_cv import (
    load_coco_catsdogs,
    load_waterbirds,
    make_model,
    resolve_coco_annotation_paths,
    set_seed,
)

DATASET_SPECS = {
    "waterbirds_seg": {
        "class_names": ["background", "landbird", "waterbird"],
        "summary_names": ("Landbird", "Waterbird"),
    },
    "coco_cd": {
        "class_names": ["background", "cat", "dog"],
        "summary_names": ("Cat", "Dog"),
    },
}


def evaluate(model, loader, device, num_classes, dataset):
    spec = DATASET_SPECS[dataset]

    # Global per-class IoU counts.
    class_inter = torch.zeros(num_classes, dtype=torch.int64)
    class_union = torch.zeros(num_classes, dtype=torch.int64)

    # Per-group per-class IoU counts for the class-specific gap summaries.
    group_class_inter = {}
    group_class_union = {}

    # Image-level uncertainty and flip statistics.
    total_binary_fg_inter = 0.0
    total_binary_fg_union = 0.0
    flip_image_sum = 0.0
    flip_image_count = 0
    pred_risk_records = []

    # Foreground error decomposition.
    fg_correct_px = 0.0
    fg_flip_px = 0.0
    fg_miss_px = 0.0
    fg_total_px = 0.0

    model.eval()
    with torch.no_grad():
        for imgs, masks, group_ids in tqdm(loader, desc="Inference"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            group_ids = group_ids.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            gt_fg = masks > 0
            pred_fg = preds > 0
            correct = gt_fg & (preds == masks)
            flip = gt_fg & pred_fg & (preds != masks)
            miss = gt_fg & (preds == 0)

            fg_total_px += gt_fg.sum().item()
            fg_correct_px += correct.sum().item()
            fg_flip_px += flip.sum().item()
            fg_miss_px += miss.sum().item()

            probs = torch.softmax(logits, dim=1)
            p1 = probs[:, 1]
            p2 = probs[:, 2]
            denom = p1 + p2
            p1_t = p1 / denom
            p2_t = p2 / denom
            entropy = -(p1_t * torch.log(p1_t) + p2_t * torch.log(p2_t))

            for batch_idx in range(preds.shape[0]):
                gt_fg_b = gt_fg[batch_idx]
                gt_fg_count = gt_fg_b.sum().item()
                flip_image_sum += flip[batch_idx].sum().item() / gt_fg_count
                flip_image_count += 1

                pred_fg_count = pred_fg[batch_idx].sum().item()
                if pred_fg_count > 0:
                    pred_fg_risk = entropy[batch_idx][pred_fg[batch_idx]].mean().item()
                    pred_risk_records.append(
                        (pred_fg_risk, flip[batch_idx].sum().item() / gt_fg_count)
                    )

            for class_id in range(num_classes):
                pred_c = preds == class_id
                target_c = masks == class_id
                class_inter[class_id] += (pred_c & target_c).sum().item()
                class_union[class_id] += (pred_c | target_c).sum().item()

            total_binary_fg_inter += (pred_fg & gt_fg).sum().item()
            total_binary_fg_union += (pred_fg | gt_fg).sum().item()

            for gid in torch.unique(group_ids.view(-1)).cpu().tolist():
                gid = int(gid)
                group_mask = group_ids == gid
                if gid not in group_class_inter:
                    group_class_inter[gid] = torch.zeros(num_classes, dtype=torch.int64)
                    group_class_union[gid] = torch.zeros(num_classes, dtype=torch.int64)

                gt_group = masks[group_mask]
                pred_group = preds[group_mask]

                for class_id in range(num_classes):
                    pred_c = pred_group == class_id
                    target_c = gt_group == class_id
                    group_class_inter[gid][class_id] += (pred_c & target_c).sum().item()
                    group_class_union[gid][class_id] += (pred_c | target_c).sum().item()

    class_iou = class_inter / class_union
    binary_iou = total_binary_fg_inter / total_binary_fg_union
    flip_img_mean = flip_image_sum / max(flip_image_count, 1)
    fg_correct_rate = fg_correct_px / fg_total_px
    fg_flip_rate = fg_flip_px / fg_total_px
    fg_miss_rate = fg_miss_px / fg_total_px

    print("Per-class IoU (global):")
    for class_id, name in enumerate(spec["class_names"]):
        print(f"  {name}: {class_iou[class_id].item():.4f}")
    print(f"Test binary foreground IoU (any label): {binary_iou:.4f}")

    g0 = (group_class_inter[0] / group_class_union[0])[1].item()
    g1 = (group_class_inter[1] / group_class_union[1])[1].item()
    g2 = (group_class_inter[2] / group_class_union[2])[2].item()
    g3 = (group_class_inter[3] / group_class_union[3])[2].item()
    class_a_name, class_b_name = spec["summary_names"]
    print(
        f"{class_a_name} IoU: group0={g0:.4f} group1={g1:.4f} gap(g0-g1)={g0 - g1:.4f}"
    )
    print(
        f"{class_b_name} IoU: group3={g3:.4f} group2={g2:.4f} gap(g3-g2)={g3 - g2:.4f}"
    )

    assert abs((fg_correct_rate + fg_flip_rate + fg_miss_rate) - 1.0) < 1e-5
    print("Foreground error decomposition:")
    print(f"  FG-Corr: {fg_correct_rate:.6f}")
    print(f"  FG-Flip: {fg_flip_rate:.6f}")
    print(f"  FG-Miss: {fg_miss_rate:.6f}")
    print(f"  Flip (global): {fg_flip_rate:.6f}")
    print(f"  Flip (image-mean): {flip_img_mean:.6f}")


    pred_risk_records.sort(key=lambda x: x[0])
    print("Pred-bin deciles:")
    n = len(pred_risk_records)
    total_flip_mass = sum(flip for _, flip in pred_risk_records)
    for decile in range(10):
        start = decile * n // 10
        end = (decile + 1) * n // 10
        bucket = pred_risk_records[start:end]
        risks, flips = zip(*bucket)
        print(
            f"  d{decile + 1}: risk={sum(risks) / len(risks):.4f} "
            f"mean_flip={sum(flips) / len(flips):.4f}"
        )
    top_decile = pred_risk_records[(9 * n) // 10 :]
    top_decile_flip = sum(flip for _, flip in top_decile)
    print(f"Top-10% flip share: {top_decile_flip / total_flip_mass:.4f}")


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
    parser.add_argument("--model-name", default="resnet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument(
        "--coco-root",
        default=None,
        help="COCO root containing train2017/val2017.",
    )
    parser.add_argument(
        "--coco-annotations-train",
        default=None,
        help="COCO annotations JSON for training.",
    )
    parser.add_argument(
        "--coco-annotations-val",
        default=None,
        help="COCO annotations JSON for validation.",
    )
    parser.add_argument(
        "--coco-annotations-test",
        default=None,
        help="COCO annotations JSON for testing.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "waterbirds_seg":
        if args.data_root is None:
            raise ValueError("--data_root is required for waterbirds_seg.")
        data = load_waterbirds(args.data_root)
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
        )

    test_loader = DataLoader(
        data["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    model = make_model(args.model_name, data["num_classes"], device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    evaluate(model, test_loader, device, data["num_classes"], args.dataset)


if __name__ == "__main__":
    main()
