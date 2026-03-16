#!/usr/bin/env bash

if [ "$1" = "coco_cd" ]; then
  shift
  python -m segshort.dataset_generation.generate_coco_cd "$@"
else
  if [ "$1" = "waterbirds_seg" ]; then
    shift
  fi
  python -m segshort.dataset_generation.generate_waterbirds "$@"
fi
