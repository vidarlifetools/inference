
import cv2

def draw_bbox(img, bbox, color=(0, 255, 0)):
    if bbox is None:
        return

    x1, y1, x2, y2 = bbox
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        color,
        1, #int(round(img.shape[0] / 200)),
        1,
    )