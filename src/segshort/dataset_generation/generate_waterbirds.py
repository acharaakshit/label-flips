# Initial Waterbirds generation followed the Group DRO script:
# https://github.com/kohpangwei/group_DRO/blob/master/dataset_scripts/generate_waterbirds.py

import argparse
import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


MANIFEST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "annotations", "waterbirds_seg")
)
SUPPORTED_CORRELATIONS = {
    "0.5": os.path.join(MANIFEST_DIR, "metadata_corr0.5.csv"),
    "0.95": os.path.join(MANIFEST_DIR, "metadata_corr0.95.csv"),
}


def crop_and_resize(source_img, target_img):
    source_width, source_height = source_img.size
    target_width, target_height = target_img.size

    if (source_width < target_width) or (source_height < target_height):
        width_resize = (target_width, int((target_width / source_width) * source_height))
        if (width_resize[0] >= target_width) and (width_resize[1] >= target_height):
            source_resized = source_img.resize(width_resize, Image.LANCZOS)
        else:
            height_resize = (int((target_height / source_height) * source_width), target_height)
            source_resized = source_img.resize(height_resize, Image.LANCZOS)
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height
    if source_aspect > target_aspect:
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        resize = (offset, 0, source_width - offset, source_height)
    else:
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        resize = (0, offset, source_width, source_height - offset)

    return source_img.crop(resize).resize((target_width, target_height), Image.LANCZOS)

def print_split_stats(df):
    for split, split_label in [(0, "train"), (1, "val"), (2, "test")]:
        split_df = df.loc[df["split"] == split, :]
        print(f"{split_label}:")
        print(f"waterbirds are {np.mean(split_df['y']):.3f} of the examples")
        print(
            "y = 0, c = 0: "
            f"{np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 0):.3f}, "
            f"n = {np.sum((split_df['y'] == 0) & (split_df['place'] == 0))}"
        )
        print(
            "y = 0, c = 1: "
            f"{np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 1):.3f}, "
            f"n = {np.sum((split_df['y'] == 0) & (split_df['place'] == 1))}"
        )
        print(
            "y = 1, c = 0: "
            f"{np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 0):.3f}, "
            f"n = {np.sum((split_df['y'] == 1) & (split_df['place'] == 0))}"
        )
        print(
            "y = 1, c = 1: "
            f"{np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 1):.3f}, "
            f"n = {np.sum((split_df['y'] == 1) & (split_df['place'] == 1))}"
        )


def render_example(row, cub_dir, places_dir, images_dir, masks_dir):
    img_path = os.path.join(cub_dir, "images", row.img_filename)
    seg_path = os.path.join(
        cub_dir, "segmentations", row.img_filename.replace(".jpg", ".png")
    )
    img_np = np.asarray(Image.open(img_path).convert("RGB"))
    seg_np = np.asarray(Image.open(seg_path).convert("RGB")) / 255

    place_path = os.path.join(places_dir, row.place_filename[1:])
    place = Image.open(place_path).convert("RGB")

    img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
    img_resized = crop_and_resize(place, img_black)
    img_masked_np = np.around(np.asarray(img_resized) * (1 - seg_np)).astype(np.uint8)
    combined_img = Image.fromarray(np.asarray(img_black) + img_masked_np)

    seg_gray = Image.open(seg_path).convert("L")
    bird_bin = (np.array(seg_gray, dtype=np.uint8) > 127).astype(np.uint8)
    mask_arr = np.zeros_like(bird_bin, dtype=np.uint8)
    mask_arr[bird_bin == 1] = 1 if row.y == 0 else 2

    output_path = os.path.join(images_dir, row.img_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_img.save(output_path)

    mask_path = os.path.join(masks_dir, row.img_filename).replace(".jpg", ".png")
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    Image.fromarray(mask_arr.astype(np.uint8), mode="L").save(mask_path)


def run(args):
    cub_dir = args.cub_dir
    places_dir = args.places_dir
    output_dir = args.output_dir
    correlation_key = f"{float(args.correlation):g}"

    if not os.path.exists(cub_dir) or not os.path.exists(places_dir):
        raise FileNotFoundError("Both --cub-dir and --places-dir must exist.")

    try:
        manifest_path = SUPPORTED_CORRELATIONS[correlation_key]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_CORRELATIONS))
        raise ValueError(
            f"Unsupported correlation {correlation_key}. "
            f"Supported correlations are: {supported}."
        ) from exc

    dataset_name = f"correlation_{correlation_key}"
    df = pd.read_csv(manifest_path)
    print_split_stats(df)

    output_subfolder = os.path.join(output_dir, dataset_name)
    images_dir = os.path.join(output_subfolder, "images")
    masks_dir = os.path.join(output_subfolder, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    shutil.copyfile(manifest_path, os.path.join(images_dir, "metadata.csv"))

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        render_example(row, cub_dir, places_dir, images_dir, masks_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-dir", required=True)
    parser.add_argument("--places-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--correlation", default=0.95)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
