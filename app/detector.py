"""Lightweight YOLO11 ONNX detector (no PyTorch) so the API fits a 512 MB instance."""
from __future__ import annotations
import ast
import numpy as np
import onnxruntime as ort
from PIL import Image


class OnnxYOLO:
    def __init__(self, path: str, imgsz: int = 640):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = {int(k): v for k, v in ast.literal_eval(meta["names"]).items()}

    def _letterbox(self, img: Image.Image):
        w, h = img.size
        r = min(self.imgsz / w, self.imgsz / h)
        nw, nh = round(w * r), round(h * r)
        canvas = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
        px, py = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return x, r, px, py

    @staticmethod
    def _nms(boxes, scores, iou_thr):
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_thr]
        return keep

    def predict(self, img: Image.Image, conf: float = 0.25, iou: float = 0.5, max_det: int = 300):
        """Return list of (xyxy, confidence, class_id) in original image pixels."""
        x, r, px, py = self._letterbox(img)
        out = self.session.run(None, {self.input_name: x})[0][0].T  # (8400, 4 + nc)
        cls_scores = out[:, 4:]
        cls = cls_scores.argmax(1)
        score = cls_scores[np.arange(len(cls)), cls]
        m = score >= conf
        if not m.any():
            return []
        b, score, cls = out[m, :4], score[m], cls[m]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                         b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        # class-aware NMS via per-class coordinate offset
        keep = self._nms(xyxy + cls[:, None] * 4096.0, score, iou)[:max_det]
        xyxy, score, cls = xyxy[keep], score[keep], cls[keep]
        w, h = img.size
        xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - px) / r).clip(0, w)
        xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - py) / r).clip(0, h)
        return [(bx.tolist(), float(s), int(c)) for bx, s, c in zip(xyxy, score, cls)]
