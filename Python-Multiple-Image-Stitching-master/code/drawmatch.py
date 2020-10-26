import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def _random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    return b, g, r
def _show_matches(image_a, image_b, keypoints_a, keypoints_b, matches):
    height_a, width_a = image_a.shape[:2]
    height_b, width_b = image_b.shape[:2]
    result = np.zeros([max(height_a, height_b), width_a + width_b, 3], dtype="uint8")
    result[:height_a, :width_a] = image_a
    result[:height_b, width_a:width_a + width_b] = image_b
    matches = matches[:500]
    for (i, j) in matches:
        point_a = (int(keypoints_a[i].pt[0]), int(keypoints_a[i].pt[1]))
        point_b = (int(keypoints_b[j].pt[0]) + width_a, int(keypoints_b[j].pt[1]))
        color = _random_color()
        cv2.circle(result, point_a, 2, color, 2)
        cv2.circle(result, point_b, 2, color, 2)
        cv2.line(result, point_a, point_b, color, 2)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.imshow(result)
    plt.show()

def _analyze(image):  # -> keypoints, descriptors
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SURF_create()
    result = sift.detectAndCompute(gray_img, None)
    del sift
    return result

def _match(keypoints_b, descriptors_b, keypoints_a, descriptors_a):
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=100)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches.append((m.trainIdx, m.queryIdx))
    if len(good_matches) > 4:
        good_keypoints_a = np.float32([keypoints_a[i].pt for (_, i) in good_matches])
        good_keypoints_b = np.float32([keypoints_b[i].pt for (i, _) in good_matches])
        homography, _ = cv2.findHomography(good_keypoints_a, good_keypoints_b,cv2.RANSAC)
        return homography, good_matches
    else:
        return None
if __name__ == '__main__':
    img1 = cv2.imread('./input/WA1.jpg')
    img2 = cv2.imread('./input/WA2.jpg')
    img1 = cv2.resize(img1, (480,640))
    img2 = cv2.resize(img2, (480, 640))
    keypoints_a, descriptors_a = _analyze(img1)
    keypoints_b, descriptors_b = _analyze(img2)
    homography, matches = _match(keypoints_a, descriptors_a, keypoints_b, descriptors_b)
    _show_matches(img1,img2,keypoints_a, keypoints_b,matches)

# import cv2
# import numpy as np
#
# img = cv2.imread('./input/WA3.jpg')
# img = cv2.resize(img,dsize=(1000,1000))
# #转换为灰度图像
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #创建一个SURF对象
# surf = cv2.xfeatures2d.SURF_create(5000)
# #SIFT对象会使用Hessian算法检测关键点，并且对每个关键点周围的区域计算特征向量。该函数返回关键点的信息和描述符
# keypoints,descriptor = surf.detectAndCompute(gray,None)
# print(type(keypoints),len(keypoints),keypoints[0])
# print(descriptor.shape)
# #在图像上绘制关键点
# img = cv2.drawKeypoints(image=img,keypoints = keypoints,outImage=img,color=(255,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# #显示图像
# result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(result)
# plt.show()