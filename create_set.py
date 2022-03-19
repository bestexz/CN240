import cv2
with open('dataset/anger.txt','r') as f:
    images = [line.strip() for line in f]

for image in images:
    img = cv2.imread("train_images/"+image)
    if img is not None:
        cv2.imwrite("train_images/"+image,img)
