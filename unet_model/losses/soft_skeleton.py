
import torch
import torch.nn.functional as F

class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=40):

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):

        if len(img.shape)==4:
            print("Good")
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):

        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

    def soft_open(self, img):
        
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):

        img1 = self.soft_open(img)
        skel = F.relu(img-img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img-img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img):

        return self.soft_skel(img)

import skimage.morphology as skm
import cv2
import numpy as np
if __name__ == "__main__":
    img = cv2.imread('Data/Al-324-185C-15min/fov1/predict_grae_retraced_train_100_512/predict_al-324-200C_15min_aligned_fov1_-1-tif-.png', cv2.IMREAD_GRAYSCALE)

    img = 1-img/255
    img = cv2.dilate(img, None, iterations=1)
    img0=img.copy()
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    skel = SoftSkeletonize(num_iter=40)
    skel_img = skel.forward(img)

    nparray= skel_img.cpu().numpy()


    img0 = skm.skeletonize(img0)*255
    print((img0))
    cv2.imshow('Original Image', (img0.astype(np.uint8)))
    cv2.imshow('Skeletonized Image', nparray[0,0,:,:]*255)
    # waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

# closing all open windows
    cv2.destroyAllWindows()