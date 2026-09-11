#!/usr/bin/env python3
"""Face box from Apple Vision, the same detector the macOS app will use.

    python scripts/face_detect.py     # self-check, no camera needed

detect_face(frame) takes a BGR numpy frame, which is what OpenCV's camera
returns, and gives back the largest face as (x, y, w, h) in pixels, or
None. Using Vision here rather than OpenCV's own detector means the crop
the webcam test measures is the crop the app will feed the model, and any
margin tuned tonight carries over to Swift unchanged.
"""
import cv2
import numpy as np
import Vision
from Foundation import NSData


def detect_face(frame):
    # ponytail: PNG round trip per frame, a few ms at 640x480. Handing Vision
    # a CVPixelBuffer is the upgrade if it ever limits the frame rate.
    ok, png = cv2.imencode(".png", frame)
    data = NSData.dataWithBytes_length_(png.tobytes(), len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    handler.performRequests_error_([request], None)
    faces = request.results() or []
    if not faces:
        return None
    area = lambda f: f.boundingBox().size.width * f.boundingBox().size.height
    bb = max(faces, key=area).boundingBox()
    # Vision gives 0-1 coordinates with the origin at the BOTTOM-left;
    # OpenCV wants pixels with the origin at the TOP-left. Forget the flip
    # and the box lands on the wrong part of the frame, with no error.
    H, W = frame.shape[:2]
    x = bb.origin.x * W
    y = (1 - bb.origin.y - bb.size.height) * H
    return int(x), int(y), int(bb.size.width * W), int(bb.size.height * H)


if __name__ == "__main__":
    from pathlib import Path

    # FER2013 train faces (never test), enlarged and pasted into one corner
    # of a grey 640x480 canvas. The detected box must land in that corner.
    files = sorted(Path("data/fer2013/train/happy").glob("*.jpg"))[:30]
    size = 192
    corners = {"kiri atas": (40, 40), "kanan bawah": (640 - size - 40, 480 - size - 40)}
    for name, (cx, cy) in corners.items():
        found = inside = 0
        for f in files:
            face = cv2.resize(cv2.imread(str(f)), (size, size), interpolation=cv2.INTER_CUBIC)
            canvas = np.full((480, 640, 3), 128, np.uint8)
            canvas[cy:cy + size, cx:cx + size] = face
            box = detect_face(canvas)
            if box:
                found += 1
                x, y, w, h = box
                mx, my = x + w / 2, y + h / 2
                inside += cx <= mx <= cx + size and cy <= my <= cy + size
        print(f"{name:11}: wajah ketemu {found}/{len(files)}, kotak di pojok yang benar {inside}/{found}")
        assert found and inside == found, "box landed outside the pasted face: coordinate flip is wrong"
    print("koordinat Vision -> OpenCV: ok")
