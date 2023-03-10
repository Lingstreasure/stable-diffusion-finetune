import cv2
import os


data_dir = "/home/d5/hz/DataSet/mat/feature_test"
names = os.listdir(data_dir)
for name in names:
    print(name)
    path = os.path.join(data_dir, name)
    elements = os.listdir(path)
    for elem in elements:
        if elem.split('.')[0].endswith("render_512"):
            img_path = os.path.join(path, elem)
            img = cv2.imread(img_path)
            x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
            y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
            # print(type(x), x.dtype, x.shape)
            x = cv2.convertScaleAbs(x)
            # print(type(x), x.dtype, x.shape)
            y = cv2.convertScaleAbs(y)
            res = cv2.addWeighted(x, 0.5, y, 0.5, 0)
            res = y
            cv2.imwrite(os.path.join(path, "edge_sobel.png"), res)
            res = cv2.Canny(img, 0, 180)
            cv2.imwrite(os.path.join(path, "edge_canny.png"), res)
            res = cv2.Laplacian(img, cv2.CV_16S)
            cv2.imwrite(os.path.join(path, "edge_laplacian.png"), res)

            # print(type(res))
            # assert 0
            
            