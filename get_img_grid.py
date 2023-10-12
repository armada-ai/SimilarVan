import cv2
import os


def gen_img_grid(img, n):
    h, w, _ = img.shape
    h_interval = h // n
    w_interval = w // n
    crop_imgs, bboxes = [], []
    for j in range(n):  # h
        start_y = j * h_interval
        for i in range(n):  # w
            start_x = i * w_interval
            end_x = (i + 1) * w_interval
            end_y = (j + 1) * h_interval
            if i == n - 1:
                end_x = max(end_x, w)
            if j == n - 1:
                end_y = max(end_y, h)
            crop_img = img[start_y: end_y + 1, start_x: end_x + 1, :]
            bbox = [start_x, start_y, end_x, end_y]
            crop_imgs.append(crop_img)
            bboxes.append(bbox)
            if i < n - 1 and j < n - 1:
                bbox = [end_x - w_interval // 2, end_y - h_interval // 2, end_x + w_interval // 2, end_y + h_interval // 2]
                crop_img = img[bbox[1]: bbox[3] + 1, bbox[0]: bbox[2] + 1, :]
                crop_imgs.append(crop_img)
                bboxes.append(bbox)
    return crop_imgs, bboxes


img_path = "data/AmazonVan/amazon_van_01_00200.jpg"
img = cv2.imread(img_path)
n = 8
crop_imgs, bboxes = gen_img_grid(img, n)
print("len(crop_imgs): ", len(crop_imgs), n * n + (n - 1) * (n - 1))
