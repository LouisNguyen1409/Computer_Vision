from ultralytics import YOLO

import cv2

model_path = './runs/segment/train/weights/best.pt'

image_path = 'dataset/images/test/1be566eccffe9561.jpg'

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for result in results:
    for mask in result.masks.data:
        mask = mask.numpy() * 255
        mask = cv2.resize(mask, (W, H))
        cv2.imwrite('./output.png', mask)