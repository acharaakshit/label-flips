"""Indoor/outdoor category mapping for COCO-Stuff."""
INDOOR_THINGS = {
    "chair", "couch", "bed", "dining table", "desk", "toilet", "sink", "refrigerator",
    "tv", "laptop", "mouse", "remote", "keyboard",
    "microwave", "oven", "toaster", "blender",
    "book", "clock", "hair drier", "toothbrush", "hair brush",
    "mirror", "vase", "teddy bear", "scissors",
}

OUTDOOR_THINGS = {
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter",
    "bench",
    "skis", "snowboard", "kite", "skateboard", "surfboard"
}


INDOOR_STUFF = {
    "blanket",
    "cabinet",
    "carpet",
    "ceiling-other", "ceiling-tile",
    "counter",
    "cupboard",
    "curtain",
    "desk-stuff",
    "door-stuff",
    "mirror-stuff",
    "pillow",
    "rug",
    "shelf",
    "stairs",
    "table",
    "textile-other",
    "towel",
    "mat",
    "window-blind"
}

NEUTRAL_STUFF = {
    "wall-brick", "wall-concrete", "wall-other", "wall-panel", "wall-stone", "wall-tile", "wall-wood",
    "floor-marble", "floor-other", "floor-stone", "floor-tile", "floor-wood", "cell phone", "house", "paper", "window-other",
}

OUTDOOR_STUFF_NATURAL = {
    "branch",
    "bush",
    "clouds",
    "dirt",
    "flower",
    "fog",
    "grass",
    "gravel",
    "ground-other",
    "hill",
    "leaves",
    "moss",
    "mountain",
    "mud",
    "plant-other",
    "river",
    "rock",
    "sand",
    "sea",
    "sky-other",
    "snow",
    "tree",
    "water-other",
    "waterdrops",
}

OUTDOOR_STUFF_BUILT = {
    "bridge",
    "building-other",
    "fence",
    "pavement",
    "platform",
    "playingfield",
    "railing",
    "railroad",
    "road",
    "roof",
    "skyscraper",
    "tent",
}


INDOOR_NAMES = INDOOR_STUFF | INDOOR_THINGS

OUTDOOR_NAMES = OUTDOOR_STUFF_BUILT | OUTDOOR_STUFF_NATURAL | OUTDOOR_THINGS


def label_stuff_categories(categories):
    """Return a mapping of category_id -> context ('indoor'/'outdoor'/None)."""
    cat_to_context = {}
    for cat in categories:
        name = cat["name"].lower()
        if name in NEUTRAL_STUFF:
            context = None
        elif name in INDOOR_NAMES:
            context = "indoor"
        elif name in OUTDOOR_NAMES:
            context = "outdoor"
        else:
            context = None
        cat_to_context[cat["id"]] = context
    return cat_to_context
