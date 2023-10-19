import numpy as np
import torch
import cv2


class GridDetector(object):
    def detect(self, image_path, n=4, verbose=False):
        assert isinstance(image_path, str)
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        h_interval = h // n
        w_interval = w // n
        new_dets = []
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
                bbox = [start_x, start_y, end_x, end_y, -1, -1]
                new_dets.append(bbox)
                if i < n - 1 and j < n - 1:
                    bbox = [end_x - w_interval // 2, end_y - h_interval // 2, 
                            end_x + w_interval // 2, end_y + h_interval // 2, -1, -1]
                    new_dets.append(bbox)
            
        new_dets = torch.from_numpy(np.array(new_dets, dtype=np.int32))
        if verbose:
            print("grid_dets: ", new_dets)
        
        return new_dets

        
if __name__ == "__main__":
    detector = GridDetector()
    img_path = "/home/ubuntu/codes/SimilarVan/data/Girl/0005.jpg"
    detector.detect(img_path, 4, verbose=True)