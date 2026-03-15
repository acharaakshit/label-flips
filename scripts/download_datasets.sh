#!/usr/bin/env bash

root="${1:?Usage: $0 /path/to/download_root}"
mkdir -p "$root"
curl -L -o "$root/CUB_200_2011.tgz" "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
curl -L -o "$root/segmentations.tgz" "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz"
curl -L -o "$root/train2017.zip" "http://images.cocodataset.org/zips/train2017.zip"
curl -L -o "$root/val2017.zip" "http://images.cocodataset.org/zips/val2017.zip"
curl -L -o "$root/annotations_trainval2017.zip" "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
curl -L -o "$root/stuff_trainval2017.zip" "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuff_trainval2017.zip"
