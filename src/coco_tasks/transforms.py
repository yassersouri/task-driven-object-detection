from typing import List, Tuple


class BBoxTransform(object):
    def __repr__(self) -> str:
        raise NotImplementedError

    def __call__(self, bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
        raise NotImplementedError


class CropBBoxToImage(BBoxTransform):
    def __repr__(self):
        return "cbti"

    def __call__(self, bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
        img_w, img_h = image_size
        x, y, width, height = tuple(bbox)

        if x < 0.0:
            width += x
            x = 0.0
        if y < 0.0:
            height += y
            y = 0.0

        if x + width >= img_w:
            width = img_w - x

        if y + height >= img_h:
            height = img_h - y

        return [x, y, width, height]


class MakeBBoxSquare(BBoxTransform):
    def __call__(self, bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        larger_dim = max(w, h)

        w_half_extra = (larger_dim - w) / 2
        h_half_extra = (larger_dim - h) / 2

        new_x = x - w_half_extra
        new_y = y - h_half_extra

        return [new_x, new_y, larger_dim, larger_dim]

    def __repr__(self) -> str:
        return "ms"


class ScaleAwarePadding(BBoxTransform):
    def __init__(self, padding: float):
        self.padding = padding

    def __call__(self, bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
        width_pad = self.padding * bbox[2]
        height_pad = self.padding * bbox[3]
        return [
            (bbox[0] - width_pad),
            (bbox[1] - height_pad),
            (bbox[2] + 2 * width_pad),
            (bbox[3] + 2 * height_pad),
        ]

    def __repr__(self) -> str:
        return "pad_sa:{padding_amount:.2f}".format(padding_amount=self.padding)


class BBoxPadding(BBoxTransform):
    """
    Expands the bounding box by a certain amount in each direction. So the width and height of the bounding box will
    increase by 2 times the padding amount.
    Padding amount can be negative for a shrinking effect.
    """

    def __init__(self, padding: int):
        self.padding = padding

    def __repr__(self) -> str:
        return "pad:{padding_amount}".format(padding_amount=self.padding)

    def __call__(self, bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
        return [
            bbox[0] - self.padding,
            bbox[1] - self.padding,
            bbox[2] + 2 * self.padding,
            bbox[3] + 2 * self.padding,
        ]


class BBoxCompose(BBoxTransform):
    def __init__(self, transforms: List[BBoxTransform]):
        self.transforms = transforms

    def __repr__(self) -> str:
        return "-".join([t.__repr__() for t in self.transforms])

    def __call__(self, bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
        for t in self.transforms:
            bbox = t.__call__(bbox, image_size)
        return bbox
