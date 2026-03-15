import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from pycocotools.coco import COCO


def resize_pair(img, mask, resize_size):
    img = img.resize((resize_size, resize_size), resample=Image.BILINEAR)
    mask = mask.resize((resize_size, resize_size), resample=Image.NEAREST)
    return img, mask


def apply_segmentation_augment(img, mask, resize_size, augment, color_jitter):
    if not augment:
        return resize_pair(img, mask, resize_size)

    if torch.rand(1).item() < 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask)

    mask_has_fg = mask.getbbox() is not None
    crop_params = None
    for _ in range(10):
        candidate = transforms.RandomResizedCrop.get_params(
            img, scale=(0.8, 1.0), ratio=(0.9, 1.1)
        )
        if not mask_has_fg:
            crop_params = candidate
            break
        crop_mask = TF.crop(mask, *candidate)
        if np.array(crop_mask, dtype=np.uint8).max(initial=0) > 0:
            crop_params = candidate
            break

    if crop_params is None and mask_has_fg:
        img, mask = resize_pair(img, mask, resize_size)
    else:
        crop_params = crop_params or transforms.RandomResizedCrop.get_params(
            img, scale=(0.8, 1.0), ratio=(0.9, 1.1)
        )
        i, j, h, w = crop_params
        size = (resize_size, resize_size)
        img = TF.resized_crop(
            img,
            i,
            j,
            h,
            w,
            size,
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        mask = TF.resized_crop(
            mask,
            i,
            j,
            h,
            w,
            size,
            interpolation=transforms.InterpolationMode.NEAREST,
        )

    if color_jitter is not None and torch.rand(1).item() < 0.3:
        img = color_jitter(img)
    return img, mask


class SharedMaskDataset(Dataset):
    def __init__(
        self,
        resize_size=512,
        num_classes=3,
        augment=True,
        bg_aug_kind="none",
        bg_aug_prob=0.5,
        bg_aug_min_frac=0.1,
        bg_aug_max_frac=0.4,
    ):
        self.resize_size = resize_size
        self.augment = augment
        self.bg_aug_kind = bg_aug_kind
        self.bg_aug_prob = bg_aug_prob
        self.bg_aug_min_frac = bg_aug_min_frac
        self.bg_aug_max_frac = bg_aug_max_frac
        self.to_tensor = transforms.ToTensor()
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        )

    def build_group_index(self, group_ids):
        self.group_ids = np.array(group_ids, dtype=np.int64)
        self.indices_by_group = {}
        for idx, gid in enumerate(self.group_ids.tolist()):
            self.indices_by_group.setdefault(int(gid), []).append(idx)

    def apply_transforms(self, img, mask):
        # If background augmentation is enabled, defer photometric jitter until after bg-aug.
        jitter = self.color_jitter if (self.bg_aug_kind or "none").lower() == "none" else None
        img, mask = apply_segmentation_augment(img, mask, self.resize_size, self.augment, jitter)
        return self.to_tensor(img), torch.from_numpy(np.array(mask, dtype=np.int64)).long()

    def apply_photometric(self, img_t):
        if not self.augment or torch.rand(1).item() > 0.3:
            return img_t
        # Tensor-safe photometric jitter in [0, 1]
        b = 1.0 + (torch.rand(1).item() * 0.4 - 0.2)
        c = 1.0 + (torch.rand(1).item() * 0.4 - 0.2)
        s = 1.0 + (torch.rand(1).item() * 0.4 - 0.2)
        img_t = TF.adjust_brightness(img_t, b)
        img_t = TF.adjust_contrast(img_t, c)
        img_t = TF.adjust_saturation(img_t, s)
        return img_t.clamp(0, 1)

    def apply_bg_aug(self, img_t, mask_t, other_sample=None):
        kind = (self.bg_aug_kind or "none").lower()
        if not self.augment or kind == "none" or torch.rand(1).item() > self.bg_aug_prob:
            return img_t
        if kind == "cutmix" and other_sample is None:
            kind = "erase"
        img_t = img_t.clone()
        height, width = mask_t.shape
        min_frac = max(0.0, min(1.0, float(self.bg_aug_min_frac)))
        max_frac = max(min_frac, min(1.0, float(self.bg_aug_max_frac)))
        h = int(height * float(torch.empty(1).uniform_(min_frac, max_frac).item()))
        w = int(width * float(torch.empty(1).uniform_(min_frac, max_frac).item()))
        top = int(torch.randint(0, height - h + 1, (1,)).item())
        left = int(torch.randint(0, width - w + 1, (1,)).item())
        rect_mask = torch.zeros_like(mask_t, dtype=torch.bool)
        rect_mask[top : top + h, left : left + w] = True
        bg_mask = mask_t == 0
        apply_mask = rect_mask & bg_mask
        if not apply_mask.any():
            return img_t
        if kind == "erase":
            if torch.rand(1).item() < 0.5:
                fill = torch.rand(3, 1, device=img_t.device)
                img_t[:, apply_mask] = fill
            else:
                noise = torch.rand_like(img_t)
                img_t[:, apply_mask] = noise[:, apply_mask]
            return img_t
        if kind == "cutmix" and other_sample is not None:
            other_img, other_mask = other_sample
            other_bg = other_mask == 0
            patch = img_t[:, top : top + h, left : left + w]
            other_patch = other_img[:, top : top + h, left : left + w]
            a_bg_patch = bg_mask[top : top + h, left : left + w]
            b_bg_patch = other_bg[top : top + h, left : left + w]
            patch_mask = a_bg_patch & b_bg_patch
            if patch_mask.float().mean().item() < 0.3:
                fill = torch.rand(3, 1, device=img_t.device)
                img_t[:, apply_mask] = fill
                return img_t
            patch[:, patch_mask] = other_patch[:, patch_mask]
            img_t[:, top : top + h, left : left + w] = patch
        return img_t

    def load_sample(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        img, mask = self.load_sample(idx)
        img_t, mask_t = self.apply_transforms(img, mask)
        group_id = int(self.group_ids[idx])
        other_sample = None
        if self.bg_aug_kind and self.bg_aug_kind.lower() == "cutmix":
            pool = self.indices_by_group.get(group_id ^ 1)
            if pool:
                other_idx = int(pool[int(torch.randint(0, len(pool), (1,)).item())])
                other_img, other_mask = self.load_sample(other_idx)
                other_sample = self.apply_transforms(other_img, other_mask)
        img_t = self.apply_bg_aug(img_t, mask_t, other_sample)
        if (self.bg_aug_kind or "none").lower() != "none":
            img_t = self.apply_photometric(img_t)
        return img_t, mask_t, group_id


class WaterbirdsDataset(SharedMaskDataset):
    def __init__(
        self,
        root,
        img_files,
        group_ids,
        resize_size=512,
        num_classes=3,
        augment=True,
        bg_aug_kind="none",
        bg_aug_prob=0.5,
        bg_aug_min_frac=0.1,
        bg_aug_max_frac=0.4,
    ):
        super().__init__(
            resize_size,
            num_classes,
            augment,
            bg_aug_kind,
            bg_aug_prob,
            bg_aug_min_frac,
            bg_aug_max_frac,
        )
        self.root = root
        self.img_files = list(img_files)
        self.build_group_index(group_ids)

    def load_sample(self, idx):
        rel = self.img_files[idx]
        img = Image.open(os.path.join(self.root, "images", rel)).convert("RGB")
        mask = Image.open(
            os.path.join(self.root, "masks", rel).replace(".jpg", ".png")
        )
        return img, mask

    def __len__(self):
        return len(self.img_files)


class CocoCatsDogsDataset(SharedMaskDataset):
    def __init__(
        self,
        annotations_path,
        images_dir,
        resize_size=512,
        augment=True,
        bg_aug_kind="none",
        bg_aug_prob=0.5,
        bg_aug_min_frac=0.1,
        bg_aug_max_frac=0.4,
    ):
        super().__init__(
            resize_size,
            3,
            augment,
            bg_aug_kind,
            bg_aug_prob,
            bg_aug_min_frac,
            bg_aug_max_frac,
        )
        self.images_dir = images_dir
        self.coco = COCO(annotations_path)
        self.img_ids = sorted(self.coco.getImgIds())
        name_to_id = {
            c["name"].lower(): c["id"] for c in self.coco.loadCats(self.coco.getCatIds())
        }
        self.cat_id_to_class = {name_to_id["cat"]: 1, name_to_id["dog"]: 2}
        self.build_group_index(
            np.array(
                [int(self.coco.loadImgs([img_id])[0]["group_id"]) for img_id in self.img_ids],
                dtype=np.int64,
            )
        )

    def __len__(self):
        return len(self.img_ids)

    def load_mask(self, img_id, height, width):
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in anns:
            class_id = self.cat_id_to_class.get(ann["category_id"])
            if class_id is None:
                continue
            ann_mask = self.coco.annToMask(ann)
            mask = np.maximum(mask, ann_mask.astype(np.uint8) * class_id)
        return mask

    def load_sample(self, idx):
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        img = Image.open(os.path.join(self.images_dir, info["file_name"])).convert("RGB")
        mask = self.load_mask(img_id, info["height"], info["width"])
        return img, Image.fromarray(mask, mode="L")
