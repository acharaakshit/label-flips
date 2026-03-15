import argparse
import json
import os
import random

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from segshort.context_categories import (
    label_stuff_categories,
    INDOOR_THINGS,
    OUTDOOR_THINGS,
)

# These parameters reproduce the released dataset distribution.
PAIR = ("cat", "dog")
CONTEXTS = ("indoor", "outdoor")
COCO_SPLITS = ("train2017", "val2017")
CONTEXT_THRESHOLD = 0.5
CONTEXT_MIN_AREA = 0.02
CORRELATION_SEED = 13
RESIZE_SIZE = 512
MIN_RESIZED_RATIO = 0.01
TRAIN_CAP_CORRELATION = 0.95
DEFAULT_VAL_PER_GROUP = 50
DEFAULT_TEST_PER_GROUP = 125
SHARED_TRAIN_ANN_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "annotations",
        "coco_cd",
        f"instances_train_corr{TRAIN_CAP_CORRELATION:g}.json",
    )
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def build_category_map(categories):
    name_to_cat = {c["name"].lower(): c for c in categories}
    return (
        {name_to_cat[PAIR[0]]["id"]: 1, name_to_cat[PAIR[1]]["id"]: 2},
        [{"id": 1, "name": name_to_cat[PAIR[0]]["name"]}, {"id": 2, "name": name_to_cat[PAIR[1]]["name"]}],
    )

def infer_train_cap_from_annotations(ann_path):
    coco = load_json(ann_path)
    name_to_id = {c["name"].lower(): c["id"] for c in coco["categories"]}
    cat_id = name_to_id[PAIR[0]]
    dog_id = name_to_id[PAIR[1]]
    img_to_classes = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid not in (cat_id, dog_id):
            continue
        img_id = ann["image_id"]
        img_to_classes.setdefault(img_id, set()).add(cid)
    cat_images = sum(1 for cls in img_to_classes.values() if cat_id in cls)
    dog_images = sum(1 for cls in img_to_classes.values() if dog_id in cls)
    return min(cat_images, dog_images)


def resolve_train_cap_per_class(train_correlation):
    if train_correlation == TRAIN_CAP_CORRELATION:
        return infer_train_cap_from_annotations(SHARED_TRAIN_ANN_PATH)
    return None


def compute_thing_context_areas(annotations, categories, exclude_names):
    name_to_id = {c["name"].lower(): c["id"] for c in categories}
    indoor_names = (INDOOR_THINGS - exclude_names) & name_to_id.keys()
    outdoor_names = (OUTDOOR_THINGS - exclude_names) & name_to_id.keys()
    cat_to_context = {name_to_id[name]: "indoor" for name in indoor_names}
    cat_to_context.update({name_to_id[name]: "outdoor" for name in outdoor_names})
    image_area = {}
    for ann in annotations:
        if ann["category_id"] not in cat_to_context:
            continue
        context = cat_to_context[ann["category_id"]]
        entry = image_area.setdefault(ann["image_id"], {"indoor": 0.0, "outdoor": 0.0})
        entry[context] += float(ann["area"])
    return image_area


def compute_image_contexts(stuff_json_paths, threshold, min_area_ratio, extra_area):
    image_area = {}
    image_info = {}
    for stuff_json_path in stuff_json_paths:
        coco = load_json(stuff_json_path)
        cat_context = label_stuff_categories(coco["categories"])
        for ann in coco["annotations"]:
            context = cat_context[ann["category_id"]]
            if context not in CONTEXTS:
                continue
            entry = image_area.setdefault(ann["image_id"], {"indoor": 0.0, "outdoor": 0.0})
            entry[context] += float(ann["area"])
        for img in coco["images"]:
            image_info[img["id"]] = (img["width"], img["height"])
            image_area.setdefault(img["id"], {"indoor": 0.0, "outdoor": 0.0})

    for img_id, areas in extra_area.items():
        entry = image_area.setdefault(img_id, {"indoor": 0.0, "outdoor": 0.0})
        entry["indoor"] += areas["indoor"]
        entry["outdoor"] += areas["outdoor"]

    image_context = {}
    for img_id, (w, h) in image_info.items():
        areas = image_area[img_id]
        indoor_area = areas["indoor"]
        outdoor_area = areas["outdoor"]
        total = indoor_area + outdoor_area
        img_area = float(w * h)
        min_area = min_area_ratio * img_area
        if total <= 0.0:
            image_context[img_id] = "unknown"
            continue
        indoor_ratio = indoor_area / total
        if indoor_ratio >= threshold and indoor_area >= min_area:
            image_context[img_id] = "indoor"
        elif indoor_ratio <= (1.0 - threshold) and outdoor_area >= min_area:
            image_context[img_id] = "outdoor"
        else:
            image_context[img_id] = "unknown"
    return image_context


def merge_splits_with_prefix(coco_root):
    first_split = COCO_SPLITS[0]
    first_coco = load_json(os.path.join(coco_root, "annotations", f"instances_{first_split}.json"))
    merged = {
        "images": [],
        "annotations": [],
        "categories": first_coco["categories"],
        "info": first_coco["info"],
        "licenses": first_coco["licenses"],
    }
    for split, coco in ((first_split, first_coco),) + tuple(
        (split, load_json(os.path.join(coco_root, "annotations", f"instances_{split}.json")))
        for split in COCO_SPLITS[1:]
    ):
        for img in coco["images"]:
            img_copy = dict(img)
            img_copy["file_name"] = os.path.join(split, img_copy["file_name"])
            merged["images"].append(img_copy)
        merged["annotations"].extend(coco["annotations"])
    return merged


def decode_annotation_mask(annotation, height, width):
    segmentation = annotation["segmentation"]
    rle = mask_utils.frPyObjects(segmentation, height, width)
    if isinstance(rle, list):
        rle = mask_utils.merge(rle)

    mask = mask_utils.decode(rle)
    return mask.astype(bool)


def has_valid_fg_bg(img_id, image_sizes, anns_by_img):
    width, height = image_sizes[img_id]
    anns = anns_by_img[img_id]
    mask = decode_annotation_mask(anns[0], height, width)
    for ann in anns[1:]:
        mask |= decode_annotation_mask(ann, height, width)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_resized = mask_img.resize((RESIZE_SIZE, RESIZE_SIZE), resample=Image.NEAREST)
    mask_arr = np.array(mask_resized) > 0
    fg_ratio = float(mask_arr.mean())
    return MIN_RESIZED_RATIO < fg_ratio < (1.0 - MIN_RESIZED_RATIO)


def build_balanced_splits(
    coco_root,
    output_dir,
    stuff_annotations_dir,
    train_correlation,
):
    correlation_map = {PAIR[0]: "indoor", PAIR[1]: "outdoor"}
    merged = merge_splits_with_prefix(coco_root)
    id_map, categories = build_category_map(merged["categories"])
    keep_cat_ids = set(id_map.keys())

    kept_annotations = []
    kept_image_ids = set()
    for ann in merged["annotations"]:
        if ann["category_id"] not in keep_cat_ids:
            continue
        ann_copy = dict(ann)
        ann_copy["category_id"] = id_map[ann["category_id"]]
        kept_annotations.append(ann_copy)
        kept_image_ids.add(ann["image_id"])

    image_sizes = {img["id"]: (img["width"], img["height"]) for img in merged["images"]}
    anns_by_img = {}
    for ann in kept_annotations:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    valid_image_ids = {
        img_id for img_id in kept_image_ids if has_valid_fg_bg(img_id, image_sizes, anns_by_img)
    }
    dropped = len(kept_image_ids) - len(valid_image_ids)
    print(f"Dropped {dropped} images due to fg/bg area constraints.")

    kept_image_ids = valid_image_ids
    kept_annotations = [ann for ann in kept_annotations if ann["image_id"] in kept_image_ids]
    kept_images = [img for img in merged["images"] if img["id"] in kept_image_ids]

    stuff_paths = [
        os.path.join(stuff_annotations_dir, f"stuff_{split}.json") for split in COCO_SPLITS
    ]

    exclude_names = {name.lower() for name in PAIR}
    thing_area = compute_thing_context_areas(
        merged["annotations"], merged["categories"], exclude_names
    )
    image_context = compute_image_contexts(
        stuff_paths, CONTEXT_THRESHOLD, CONTEXT_MIN_AREA, extra_area=thing_area
    )
    unknown_count = sum(
        1 for img_id in kept_image_ids if image_context[img_id] == "unknown"
    )
    print(f"Unknown context images (kept): {unknown_count} / {len(kept_image_ids)}")

    id_to_name = {cat["id"]: cat["name"].lower() for cat in categories}
    class_to_images = {
        name: {"indoor": set(), "outdoor": set()} for name in id_to_name.values()
    }
    image_to_classes = {}
    for ann in kept_annotations:
        class_name = id_to_name[ann["category_id"]]
        image_id = ann["image_id"]
        image_to_classes.setdefault(image_id, set()).add(class_name)
        context = image_context[image_id]
        if context in CONTEXTS:
            class_to_images[class_name][context].add(image_id)

    group_pools = {}
    for class_name in PAIR:
        for context in CONTEXTS:
            ids = class_to_images[class_name][context]
            group_pools[(class_name, context)] = {
                img_id
                for img_id in ids
                if image_to_classes[img_id] == {class_name}
            }

    print("Available subgroup image counts:")
    for class_name in PAIR:
        for context in CONTEXTS:
            count = len(group_pools[(class_name, context)])
            print(f"  {class_name}_{context}: {count}")

    group_id_map = {}
    for class_name in PAIR:
        y = 0 if class_name == PAIR[0] else 1
        for context in CONTEXTS:
            spurious = 0 if context == "indoor" else 1
            for img_id in group_pools[(class_name, context)]:
                group_id_map[img_id] = y * 2 + spurious

    rng = random.Random(CORRELATION_SEED)
    val_ids = set()
    test_ids = set()
    for class_name in PAIR:
        for context in CONTEXTS:
            pool = sorted(group_pools[(class_name, context)])
            need = DEFAULT_VAL_PER_GROUP + DEFAULT_TEST_PER_GROUP
            if len(pool) < need:
                raise ValueError(
                    f"Not enough images for {class_name}/{context}: "
                    f"need {need}, found {len(pool)}."
                )
            picks = rng.sample(pool, need)
            val_ids.update(picks[:DEFAULT_VAL_PER_GROUP])
            test_ids.update(picks[DEFAULT_VAL_PER_GROUP:])

    remaining_ids = {
        img_id
        for img_id in kept_image_ids
        if img_id not in val_ids and img_id not in test_ids
    }

    # Build correlated train from remaining IDs
    per_class_limits = {}
    for class_name in PAIR:
        major_context = correlation_map[class_name]
        minor_context = "outdoor" if major_context == "indoor" else "indoor"
        major_pool = group_pools[(class_name, major_context)] & remaining_ids
        minor_pool = group_pools[(class_name, minor_context)] & remaining_ids
        if not major_pool or not minor_pool:
            raise ValueError(
                f"Insufficient train images for {class_name} after val/test sampling."
            )
        max_n = min(
            len(major_pool) / train_correlation,
            len(minor_pool) / (1.0 - train_correlation),
        )
        per_class_limits[class_name] = (int(max_n), major_pool, minor_pool)

    target_n = min(limit[0] for limit in per_class_limits.values())
    train_cap_per_class = resolve_train_cap_per_class(train_correlation)
    if train_cap_per_class is not None:
        target_n = min(target_n, train_cap_per_class)
    if target_n <= 0:
        raise ValueError("Not enough images to build requested train correlation.")

    train_ids = set()
    for class_name in PAIR:
        major_context = correlation_map[class_name]
        minor_context = "outdoor" if major_context == "indoor" else "indoor"
        _, major_pool, minor_pool = per_class_limits[class_name]
        n_major = int(target_n * train_correlation)
        n_minor = target_n - n_major
        n_major = max(1, min(n_major, len(major_pool)))
        n_minor = max(1, min(n_minor, len(minor_pool)))
        train_ids.update(rng.sample(sorted(major_pool), n_major))
        train_ids.update(rng.sample(sorted(minor_pool), n_minor))

    train_tag = f"train_corr{train_correlation:g}"
    outputs = []
    for split_name, split_tag, ids in (
        ("train", train_tag, train_ids),
        ("val", "val_balanced", val_ids),
        ("test", "test_balanced", test_ids),
    ):
        images = [img for img in kept_images if img["id"] in ids]
        annotations = [ann for ann in kept_annotations if ann["image_id"] in ids]
        for img in images:
            img["group_id"] = group_id_map[img["id"]]
        out_ann_path = os.path.join(output_dir, "annotations", f"instances_{split_tag}.json")
        payload = {
            "info": merged["info"],
            "licenses": merged["licenses"],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        write_json(out_ann_path, payload)
        outputs.append((split_name, out_ann_path, len(images), len(annotations)))

    for label, ids in (("Train", train_ids), ("Val", val_ids), ("Test", test_ids)):
        print(f"{label} subgroup counts:")
        for class_name in PAIR:
            for context in CONTEXTS:
                print(
                    f"  {class_name}_{context}: "
                    f"{len(group_pools[(class_name, context)] & ids)}"
                )
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Generate MS COCO subset for a confusable class pair."
    )
    parser.add_argument("--coco-root", required=True, help="Path to COCO root.")
    parser.add_argument(
        "--output-dir", required=True, help="Where to write filtered annotations."
    )
    parser.add_argument(
        "--stuff-annotations-dir",
        required=True,
        help="Path to COCO-Stuff annotations (expects stuff_<split>.json).",
    )
    parser.add_argument(
        "--train-correlation",
        type=float,
        required=True,
        help="Correlation level for train split.",
    )
    args = parser.parse_args()

    outputs = build_balanced_splits(
        args.coco_root,
        args.output_dir,
        args.stuff_annotations_dir,
        args.train_correlation,
    )
    for split, ann_path, num_images, num_annotations in outputs:
        print(
            f"{split} [balanced]: {num_images} images, {num_annotations} annotations -> {ann_path}"
        )


if __name__ == "__main__":
    main()
