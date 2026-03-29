"""
YOLOv8-n style skeleton implemented in Flax/Linen for inference and bounding-box generation.

This is a practical, compact reimplementation aimed at producing usable bounding boxes
for images. It's not an official reproduction — it's an engineering-faithful skeleton
(Conv, C2f, SPPF, simple FPN/PAN-like neck, decoupled head, and postprocessing).

Usage (inference only):
  - Initialize the model variables with a dummy input and load pretrained weights if you
    have them (checkpoint loading not included here).
  - Call `apply_fn(params, images)` to get raw outputs per scale.
  - Use `decode_outputs` + `nms` to get final boxes.

Notes:
  - Activation used: SiLU (nn.swish alias). BatchNorm uses use_running_average flag in apply.
  - This code is intended as a starting point for adaptation to your dataset and training.

Requirements:
  - jax, flax, numpy, PIL (for image I/O), and optionally OpenCV for preprocessing.

"""

from typing import Tuple, Sequence, List, Any
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from functools import partial

# ------------------------- Basic building blocks -------------------------

class Conv(nn.Module):
    out_ch: int
    k: int = 1
    s: int = 1

    @nn.compact
    def __call__(self, x, train: bool = False):
        x = nn.Conv(self.out_ch, kernel_size=(self.k, self.k), strides=(self.s, self.s), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.swish(x)

class DWConv(nn.Module):
    out_ch: int
    k: int = 3
    s: int = 1

    @nn.compact
    def __call__(self, x, train: bool = False):
        in_ch = x.shape[-1]
        x = nn.Conv(in_ch, kernel_size=(self.k, self.k), strides=(self.s, self.s), padding='SAME', feature_group_count=in_ch, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.swish(x)
        x = nn.Conv(self.out_ch, kernel_size=(1, 1), strides=(1, 1), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.swish(x)

class C2f(nn.Module):
    out_ch: int
    n: int = 1

    @nn.compact
    def __call__(self, x, train: bool = False):
        # simplified C2f: split -> sequence of convs -> concat -> conv
        hidden_ch = self.out_ch // 2
        x1 = nn.Conv(hidden_ch, (1,1), strides=(1,1), padding='SAME', use_bias=False)(x)
        x1 = nn.BatchNorm(use_running_average=not train)(x1)
        x1 = nn.swish(x1)

        y = x1
        for i in range(self.n):
            y = Conv(hidden_ch, k=3, s=1)(y, train=train)

        out = jnp.concatenate([y, x1], axis=-1)
        out = Conv(self.out_ch, k=1, s=1)(out, train=train)
        return out

class SPPF(nn.Module):
    out_ch: int
    k: int = 5

    @nn.compact
    def __call__(self, x, train: bool = False):
        # Spatial Pyramid Pooling - Fast variant (stacked maxpools)
        x = Conv(self.out_ch//2, k=1)(x, train=train)
        y1 = nn.max_pool(x, window_shape=(self.k, self.k), strides=(1,1), padding='SAME')
        y2 = nn.max_pool(y1, window_shape=(self.k, self.k), strides=(1,1), padding='SAME')
        out = jnp.concatenate([x, y1, y2], axis=-1)
        out = Conv(self.out_ch, k=1)(out, train=train)
        return out

# ------------------------- Backbone -------------------------

class YOLOv8Backbone(nn.Module):
    # channels for yolov8n-ish
    chs: Sequence[int] = (32, 64, 128, 256)

    @nn.compact
    def __call__(self, x, train: bool = False):
        # stem
        x = Conv(self.chs[0], k=3, s=2)(x, train=train)  # /2

        c1 = C2f(self.chs[1], n=1)(x, train=train)       # small stage
        c2 = C2f(self.chs[2], n=2)(c1, train=train)      # medium stage
        c3 = C2f(self.chs[3], n=2)(c2, train=train)      # deeper

        p5 = SPPF(self.chs[3])(c3, train=train)          # last feature

        # Return three scales (P3, P4, P5-like)
        return c1, c2, p5

# ------------------------- Neck (FPN+PAN simplified) -------------------------

class Upsample(nn.Module):
    scale: int = 2

    def __call__(self, x):
        bs, h, w, c = x.shape
        return jax.image.resize(x, (bs, h * self.scale, w * self.scale, c), method='nearest')

class DownsampleConv(nn.Module):
    out_ch: int

    @nn.compact
    def __call__(self, x, train: bool = False):
        return Conv(self.out_ch, k=3, s=2)(x, train=train)

class YOLOv8Neck(nn.Module):
    chs: Sequence[int] = (64, 128, 256)

    @nn.compact
    def __call__(self, feats: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], train: bool = False):
        c1, c2, p5 = feats  # low, mid, high

        # lateral convs
        p5_ = Conv(self.chs[2], k=1)(p5, train=train)
        up = Upsample()(p5_)

        p4 = jnp.concatenate([up, c2], axis=-1)
        p4 = C2f(self.chs[1], n=2)(p4, train=train)

        up2 = Upsample()(p4)
        p3 = jnp.concatenate([up2, c1], axis=-1)
        p3 = C2f(self.chs[0], n=2)(p3, train=train)

        # bottom-up path
        p3_down = DownsampleConv(self.chs[1])(p3, train=train)
        p4b = jnp.concatenate([p3_down, p4], axis=-1)
        p4b = C2f(self.chs[1], n=2)(p4b, train=train)

        p4_down = DownsampleConv(self.chs[2])(p4b, train=train)
        p5b = jnp.concatenate([p4_down, p5_], axis=-1)
        p5b = C2f(self.chs[2], n=2)(p5b, train=train)

        # outputs: small, medium, large (P3, P4b, P5b)
        return p3, p4b, p5b

# ------------------------- Head (decoupled) -------------------------

class DetectionHead(nn.Module):
    num_classes: int
    anchors: Sequence[int] = (8, 16, 32)  # relative stride anchors (informal)
    ch: int = 64

    @nn.compact
    def __call__(self, feats: Sequence[jnp.ndarray], train: bool = False):
        outputs = []
        for i, x in enumerate(feats):
            # shared conv
            x = Conv(self.ch, k=3)(x, train=train)

            # regression branch
            reg = Conv(self.ch, k=3)(x, train=train)
            reg = Conv(4, k=1)(reg, train=train)  # tx, ty, tw, th

            # objectness
            obj = Conv(self.ch, k=3)(x, train=train)
            obj = Conv(1, k=1)(obj, train=train)

            # classification
            cls = Conv(self.ch, k=3)(x, train=train)
            cls = Conv(self.num_classes, k=1)(cls, train=train)

            # concat channels: [reg(4), obj(1), cls(C)]
            out = jnp.concatenate([reg, obj, cls], axis=-1)
            outputs.append(out)
        return outputs

# ------------------------- Full model -------------------------

class YOLOv8Flax(nn.Module):
    num_classes: int = 1  # for airplanes, set to 1

    @nn.compact
    def __call__(self, x, train: bool = False):
        backbone = YOLOv8Backbone()(x, train=train)
        neck_feats = YOLOv8Neck()(backbone, train=train)
        head_outs = DetectionHead(num_classes=self.num_classes)(neck_feats, train=train)
        return head_outs

# ------------------------- Post-processing utilities -------------------------

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def make_grid(h, w):
    # returns grid of shape (h, w, 2) with center coordinates
    xs = jnp.arange(w)
    ys = jnp.arange(h)
    gx, gy = jnp.meshgrid(xs, ys)
    grid = jnp.stack([gx, gy], axis=-1)
    return grid  # y,x ordering: grid[y,x] = [x,y]

def decode_outputs(outputs: Sequence[jnp.ndarray], strides: Sequence[int], conf_thresh: float = 0.25):
    # outputs: list of tensors [B,H,W,4+1+C]
    # strides: stride for each scale (e.g., [8,16,32]) relative to input
    boxes_all = []
    scores_all = []
    classes_all = []

    for out, s in zip(outputs, strides):
        # out shape (B,H,W, D)
        b, h, w, d = out.shape
        C = d - 5
        grid = make_grid(h, w)
        grid = jnp.expand_dims(grid, 0)  # (1,H,W,2)

        # split
        reg = out[..., :4]
        obj = sigmoid(out[..., 4:5])
        cls_logits = out[..., 5:]
        cls_prob = sigmoid(cls_logits)

        # box decode (simple): tx,ty,tw,th -> cx = (tx + gx) * stride
        # note: this is a simplified decode; in YOLOv8 there are more details
        tx = reg[..., 0]
        ty = reg[..., 1]
        tw = reg[..., 2]
        th = reg[..., 3]

        cx = (sigmoid(tx) + grid[..., 0]) * s
        cy = (sigmoid(ty) + grid[..., 1]) * s
        bw = jnp.exp(tw) * s
        bh = jnp.exp(th) * s

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # flatten
        x1 = x1.reshape(-1)
        y1 = y1.reshape(-1)
        x2 = x2.reshape(-1)
        y2 = y2.reshape(-1)

        # compute scores and classes
        obj = obj.reshape(-1)
        cls_prob = cls_prob.reshape(-1, C)
        cls_scores = jnp.max(cls_prob, axis=-1)
        cls_ids = jnp.argmax(cls_prob, axis=-1)

        scores = obj * cls_scores

        # threshold
        mask = scores > conf_thresh
        if mask.sum() == 0:
            boxes_all.append(jnp.zeros((0,4)))
            scores_all.append(jnp.zeros((0,)))
            classes_all.append(jnp.zeros((0,), dtype=jnp.int32))
            continue

        boxes = jnp.stack([x1, y1, x2, y2], axis=-1)
        boxes = boxes[mask]
        scores = scores[mask]
        classes = cls_ids[mask]

        boxes_all.append(boxes)
        scores_all.append(scores)
        classes_all.append(classes)

    # concat all scales
    if len(boxes_all) == 0:
        return jnp.zeros((0,4)), jnp.zeros((0,)), jnp.zeros((0,), dtype=jnp.int32)

    boxes = jnp.concatenate(boxes_all, axis=0)
    scores = jnp.concatenate(scores_all, axis=0)
    classes = jnp.concatenate(classes_all, axis=0)

    return boxes, scores, classes

# Simple NMS (CPU via numpy for convenience)

def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45, max_boxes: int = 100):
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0 and len(keep) < max_boxes:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)

# ------------------------- High-level inference helper -------------------------

def preprocess_image_pil(img: Any, target_size: int = 640):
    # img: PIL.Image or numpy array HxWxC
    from PIL import Image
    if isinstance(img, Image.Image):
        im = img.convert('RGB')
        im = im.resize((target_size, target_size))
        arr = np.asarray(im).astype(np.float32) / 255.0
    else:
        # assume numpy array
        im = img
        im = np.asarray(im)
        # resize
        from PIL import Image
        im = Image.fromarray(im.astype('uint8')).resize((target_size, target_size))
        arr = np.asarray(im).astype(np.float32) / 255.0

    # HWC -> CHW and add batch dim
    arr = arr.astype(np.float32)
    return arr[None, ...]

def postprocess_and_nms(boxes_j: jnp.ndarray, scores_j: jnp.ndarray, classes_j: jnp.ndarray, image_shape: Tuple[int,int], iou_thresh: float = 0.45, max_det: int = 100):
    boxes = np.array(boxes_j)
    scores = np.array(scores_j)
    classes = np.array(classes_j)

    if boxes.shape[0] == 0:
        return []

    keep = nms_numpy(boxes, scores, iou_thresh=iou_thresh, max_boxes=max_det)
    out = []
    H, W = image_shape
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        score = float(scores[i])
        cls = int(classes[i])
        # clamp
        x1 = max(0.0, x1); y1 = max(0.0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        out.append({'bbox':[float(x1), float(y1), float(x2 - x1), float(y2 - y1)], 'score': score, 'class': cls})
    return out

# ------------------------- Example usage stub -------------------------

if __name__ == '__main__':
    # quick demo showing shape flow
    import numpy as np

    model = YOLOv8Flax(num_classes=1)

    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 640, 640, 3), dtype=jnp.float32)

    variables = model.init(rng, dummy, train=False)
    params = variables['params']
    # apply
    outs = model.apply(variables, dummy, train=False)

    # outs is list of three tensors [B,H,W,D]
    for o in outs:
        print('out shape', o.shape)

    # decode example (strides chosen heuristically for this skeleton)
    boxes, scores, classes = decode_outputs(outs, strides=[8,16,32], conf_thresh=0.25)
    print('decoded', boxes.shape, scores.shape, classes.shape)

    # no trained weights here; results are meaningless but pipeline works

# End of file
