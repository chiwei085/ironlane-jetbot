import time

import numpy as np

from .utils import clip

CLS_BLOCKED = 0
CLS_CAR = 1
CLS_OBSTACLE = 2
CLS_PEDESTRIAN = 3
CLS_RAIL = 4
CLS_STOP = 5


def decode_trt_yolov8(outputs, conf_th=0.5):
    """Decode TensorRT YOLOv8 output into boxes, confidences, and class ids.

    Args:
        outputs: List of model outputs from engine.infer(x).
        conf_th: Confidence threshold for filtering.

    Returns:
        Tuple (boxes_xyxy, confs, cls_ids).
    """
    out = np.asarray(outputs[0])

    if out.ndim != 3:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    b, c, n = out.shape
    if b != 1 or c <= 4:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    num_classes = c - 4

    pred = out[0].transpose(1, 0)  # (N, C)
    xywh = pred[:, :4]  # (N,4)
    cls_logits = pred[:, 4 : 4 + num_classes]  # (N, num_classes)

    cls_ids = np.argmax(cls_logits, axis=1).astype(np.int32)
    row_idx = np.arange(cls_logits.shape[0], dtype=np.int64)
    cls_max = cls_logits[row_idx, cls_ids]
    conf = cls_max.astype(np.float32, copy=False)

    keep = conf >= float(conf_th)
    if not keep.any():
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    xywh = xywh[keep]
    conf = conf[keep]
    cls_ids = cls_ids[keep]

    boxes = np.empty_like(xywh, dtype=np.float32)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0  # x1
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0  # y1
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0  # x2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0  # y2
    return boxes, conf, cls_ids


def box_iou(box, boxes):
    """Compute IoU between one box and many boxes.

    Args:
        box: Single box [x1, y1, x2, y2].
        boxes: Array of boxes with shape (N, 4).

    Returns:
        IoU array of shape (N,).
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - inter_area + 1e-6
    return inter_area / union


def nms_single_class(boxes, scores, iou_th=0.5):
    """Perform NMS on boxes for a single class.

    Args:
        boxes: Array of boxes (N, 4).
        scores: Confidence scores (N,).
        iou_th: IoU threshold for suppression.

    Returns:
        Indices of kept boxes.
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int32)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i], boxes[order[1:]])
        inds = np.where(ious < float(iou_th))[0]
        order = order[1:][inds]

    return np.asarray(keep, dtype=np.int32)


def nms_per_class(boxes, confs, cls_ids, iou_th=0.5):
    """Run class-wise NMS and return filtered detections.

    Args:
        boxes: Array of boxes (N, 4).
        confs: Confidence scores (N,).
        cls_ids: Class ids (N,).
        iou_th: IoU threshold.

    Returns:
        Tuple (boxes, confs, cls_ids) after NMS.
    """
    if boxes.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    keep_all = []
    for cls in np.unique(cls_ids):
        idxs = np.where(cls_ids == cls)[0]
        if idxs.size == 0:
            continue
        b = boxes[idxs]
        s = confs[idxs]
        keep = nms_single_class(b, s, iou_th=iou_th)
        keep_all.append(idxs[keep])

    if not keep_all:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    keep_all = np.concatenate(keep_all)
    return boxes[keep_all], confs[keep_all], cls_ids[keep_all]


def _pick_nearest_by_width(boxes_xyxy, cls_ids, in_w, alert_width_by_cls):
    """Pick the widest box per alert width threshold.

    Args:
        boxes_xyxy: Array of boxes (N, 4).
        cls_ids: Class ids (N,).
        in_w: Input width for center normalization.
        alert_width_by_cls: Dict of minimum widths per class.

    Returns:
        Tuple (cls_id, box, box_width, cx_norm) or None.
    """
    best_i = -1
    best_w = 0.0

    n = int(boxes_xyxy.shape[0])
    for i in range(n):
        cid = int(cls_ids[i])
        x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[i]]
        bw = x2 - x1
        thr = float(alert_width_by_cls.get(cid, 50.0))
        if bw < thr:
            continue
        if bw > best_w:
            best_w = bw
            best_i = i

    if best_i < 0:
        return None

    cid = int(cls_ids[best_i])
    x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[best_i]]
    bw = float(x2 - x1)
    cx = 0.5 * (x1 + x2)
    cx_norm = (cx - (float(in_w) * 0.5)) / (float(in_w) * 0.5)  # [-1,1]
    cx_norm = clip(cx_norm, -1.0, 1.0)
    return cid, (x1, y1, x2, y2), bw, cx_norm


def yolo_outputs_to_policy(
    outs,
    ts=None,
    in_shape=None,
    *,
    conf_th=0.5,
    iou_th=0.5,
    alert_width=50.0,
    alert_width_by_cls=None,
    # durations (seconds)
    pedestrian_slow_s=2.0,
    rail_stop_s=5.0,
    stop_sign_s=2.0,
    # avoid (two parameter sets)
    car_avoid_s=0.45,
    obstacle_avoid_s=0.45,
    car_avoid_strength=0.35,
    obstacle_avoid_strength=0.35,
    # sign convention fixer
    turn_sign=1.0,
):
    """Convert YOLO outputs into a speed/avoidance policy dict.

    Args:
        outs: Output list from engine.infer(x).
        ts: Optional timestamp (seconds).
        in_shape: Input shape (N, C, H, W) for center normalization.
        conf_th: Confidence threshold for filtering.
        iou_th: IoU threshold for NMS.
        alert_width: Default minimum width for triggering actions.
        alert_width_by_cls: Optional per-class width thresholds.
        pedestrian_slow_s: Duration for slow-down on pedestrian.
        rail_stop_s: Duration for stop on rail.
        stop_sign_s: Duration for stop sign.
        car_avoid_s: Avoid duration for cars.
        obstacle_avoid_s: Avoid duration for obstacles.
        car_avoid_strength: Avoid bias strength for cars.
        obstacle_avoid_strength: Avoid bias strength for obstacles.
        turn_sign: Sign convention multiplier for avoid direction.

    Returns:
        Policy dict with "kind" and timing/avoid fields.
    """
    now = time.time() if ts is None else float(ts)

    if in_shape is None or len(in_shape) != 4:
        in_w = 224
    else:
        in_w = int(in_shape[3])

    if alert_width_by_cls is None:
        alert_width_by_cls = {}
    # Fill defaults for missing class thresholds.
    for cid in range(6):
        if cid not in alert_width_by_cls:
            alert_width_by_cls[cid] = float(alert_width)

    boxes, confs, cls_ids = decode_trt_yolov8(outs, conf_th=conf_th)
    if boxes.shape[0] == 0:
        return {"kind": "normal", "until_ts": 0.0}

    boxes, confs, cls_ids = nms_per_class(boxes, confs, cls_ids, iou_th=iou_th)
    if boxes.shape[0] == 0:
        return {"kind": "normal", "until_ts": 0.0}

    picked = _pick_nearest_by_width(boxes, cls_ids, in_w, alert_width_by_cls)
    if picked is None:
        return {"kind": "normal", "until_ts": 0.0}

    cid, (x1, y1, x2, y2), bw, cx_norm = picked

    # blocked: permanent halt
    if cid == CLS_BLOCKED:
        return {
            "kind": "halt",
            "until_ts": float("inf"),
            "cls_id": cid,
            "box": [x1, y1, x2, y2],
            "box_w": bw,
            "cx_norm": cx_norm,
        }

    # pedestrian: slow down then resume
    if cid == CLS_PEDESTRIAN:
        return {
            "kind": "slow",
            "until_ts": now + float(pedestrian_slow_s),
            "cls_id": cid,
            "box": [x1, y1, x2, y2],
            "box_w": bw,
            "cx_norm": cx_norm,
        }

    # rail: stop then resume previous speed
    if cid == CLS_RAIL:
        return {
            "kind": "stop",
            "until_ts": now + float(rail_stop_s),
            "resume_speed": None,
            "cls_id": cid,
            "box": [x1, y1, x2, y2],
            "box_w": bw,
            "cx_norm": cx_norm,
        }

    # stop sign: brief stop then resume
    if cid == CLS_STOP:
        return {
            "kind": "stop",
            "until_ts": now + float(stop_sign_s),
            "resume_speed": None,
            "cls_id": cid,
            "box": [x1, y1, x2, y2],
            "box_w": bw,
            "cx_norm": cx_norm,
        }

    # car / obstacle: avoid with turn bias
    if cid == CLS_CAR or cid == CLS_OBSTACLE:
        # cx_norm < 0 => object on left, bias to dodge right.
        # Convention: turn<0 dodges right, turn>0 dodges left.
        desired = -1.0 if cx_norm < 0.0 else +1.0
        desired *= float(turn_sign)

        if cid == CLS_CAR:
            return {
                "kind": "avoid",
                "until_ts": now + float(car_avoid_s),
                "avoid_turn": desired,
                "avoid_strength": float(car_avoid_strength),
                "avoid_tag": "car",
                "cls_id": cid,
                "box": [x1, y1, x2, y2],
                "box_w": bw,
                "cx_norm": cx_norm,
            }

        return {
            "kind": "avoid",
            "until_ts": now + float(obstacle_avoid_s),
            "avoid_turn": desired,
            "avoid_strength": float(obstacle_avoid_strength),
            "avoid_tag": "obstacle",
            "cls_id": cid,
            "box": [x1, y1, x2, y2],
            "box_w": bw,
            "cx_norm": cx_norm,
        }

    return {"kind": "normal", "until_ts": 0.0, "cls_id": cid}
