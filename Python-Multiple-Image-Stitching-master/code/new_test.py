import numpy as np
import cv2
import sys
import os
import math
from cv2 import Stitcher
import matplotlib.pyplot as plt

# 图像匹配
# 提取特征、特征匹配、构造单应性矩阵
class matchers:
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()  # 特征获取，使用SURF算法
        # FLANN是快速最近邻搜索包，对大数据集和高维特征进行最近邻搜索的算法的集合，使用FLANN匹配，需要传入两个字典作为参数
        index_params = dict(algorithm=0, trees=5)  # 第一个：indexParams，配置要使用的算法
        search_params = dict(checks=100)  # 第二个：SearchParams，用来指定递归遍历的次数
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 提取图像特征
    def get_SURF_features(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用Hessian算法检测关键点，并且对每个关键点周围的区域计算特征向量，返回关键点的信息和描述符
        keypoints, descriptor = self.surf.detectAndCompute(image_gray, None)
        return {'keypoints': keypoints, 'descriptor': descriptor}

    # 进行图像匹配
    def match(self, image1, image2):
        # 获取两幅图片的特征
        fea_image1 = self.get_SURF_features(image1)
        fea_image2 = self.get_SURF_features(image2)
        #img4 = cv2.drawKeypoints(image=image1, keypoints=fea_image1['keypoints'], outImage=image1)
        #cv2.imwrite('KEYpoints.jpg', img4)
        # 对两幅图片的特征进行匹配
        # knnMatch：给定查询集合中的每个特征描述子，寻找k个最佳匹配
        # matches得到许多组两两匹配的关键点
        matches = self.flann.knnMatch(fea_image2['descriptor'], fea_image1['descriptor'], k=2)
        good_matches = []
        #good = [] # 纠正后的所有匹配点对
        for i, (m, n) in enumerate(matches):
            # 检测出的匹配点可能有一些是错误正例。抛弃距离大于0.7的值，则可以避免一些错误匹配
            if m.distance < 0.7 * n.distance:
                # trainIdx：测试图像的特征点下标
                # queryIdx：样本图像的特征点下标
                #good.append((m, n))
                good_matches.append((m.trainIdx, m.queryIdx))
        # 绘制匹配点对图
        #img_match_points = cv2.drawMatchesKnn(image1, fea_image1['keypoints'], image2, fea_image2['keypoints'], good,None,flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        #cv2.imwrite('drawmatch.jpg', img_match_points)
        if len(good_matches) > 4:
            points_1 = fea_image1['keypoints']
            points_2 = fea_image2['keypoints']
            matched_Points_Prev = np.float32([points_1[i].pt for (i, __) in good_matches])
            matched_Points_Current = np.float32([points_2[i].pt for (__, i) in good_matches])
            # matchedPointsCurrent：源平面中点的坐标矩阵
            # matchedPointsPrev：目标平面中点的坐标矩阵
            # RANSAC:计算单应矩阵所使用的方法-基于RANSAC的鲁棒算法
            # 第四个参数是误差阈值
            H, status = cv2.findHomography(matched_Points_Current, matched_Points_Prev, cv2.RANSAC, 4)
            return H
        else:
            return None

class Stitch_0:
    def __init__(self, path, size):
        self.path = path
        filenames = os.listdir(path)
        print('参与拼接的图片：')
        #print(filenames)
        self.images = []  # 参与拼接的图片
        for each in filenames:  # 把参与拼接的图片尺寸调到一致
            file_path = path + each
            print(file_path)
            input_image = cv2.imread(file_path)
            resize_shape = size
            input_image = cv2.resize(input_image, (resize_shape[0], resize_shape[1]))
            #input_image = self.cylindricalProjection(input_image)
            self.images.append(input_image)
        self.count = len(self.images)
        print('numbers of images: ', self.count)
        self.matcher_obj = matchers()
        self.left_list, self.right_list, self.center_im = [], [], None
        self.matcher_obj = matchers()
        self.prepare_lists()

    def cylindricalProjection(self, img):
        rows = img.shape[0]
        cols = img.shape[1]

        # f = cols / (2 * math.tan(np.pi / 8))
        result = np.zeros_like(img)
        center_x = int(cols / 2)
        center_y = int(rows / 2)
        alpha = math.pi / 4
        f = cols / (2 * math.tan(alpha / 2))
        for y in range(rows):
            for x in range(cols):
                theta = math.atan((x - center_x) / f)
                point_x = int(f * math.tan((x - center_x) / f) + center_x)
                point_y = int((y - center_y) / math.cos(theta) + center_y)

                if point_x >= cols or point_x < 0 or point_y >= rows or point_y < 0:
                    pass
                else:
                    result[y, x, :] = img[point_y, point_x, :]
        return result

    def prepare_lists(self):
        self.centerIdx = self.count / 2
        #print("Center index image : %d" % self.centerIdx)
        self.center_im = self.images[int(self.centerIdx)]
        for i in range(self.count):
            if (i <= self.centerIdx):
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])

    def leftshift(self):
        a = self.left_list[0]  # 第一张图
        for b in self.left_list[1:]:  # 左边部分剩下的图
            H = self.matcher_obj.match(a, b)  # 两张图片进行特征匹配后的单应矩阵
            print("Homography is : ", H)
            xh = np.linalg.inv(H)  # 求逆
            print("Inverse Homography :", xh)  # 逆单应矩阵
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))  # 逆单应矩阵和第一张图相乘
            ds = ds / ds[-1]
            #print("final ds=>", ds)
            f1 = np.dot(xh, np.array([0, 0, 1]))  # 获得逆单应矩阵的第三列，即xh[2][0]~xh[2][2]
            f1 = f1 / f1[-1]
            print(f1[0])
            xh[0][-1] += abs(f1[0])  # 把f1[0]的绝对值赋给xh[0][-1],即改变xh[0][2]
            xh[1][-1] += abs(f1[1])  # 改变xh[1][2]
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))  # 改变后的逆单应矩阵与第一张图相乘
            offsety = abs(int(f1[1]))
            offsetx = abs(int(f1[0]))
            dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
            print("image dsize =>", dsize)
            tmp = cv2.warpPerspective(a, xh, dsize)  # 对第一张图a进行透视变换，xh是转换矩阵，即得到拼接图片在被拼接图片的视角
            # result = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            # plt.imshow(result)
            # plt.show()
            tmp[offsety:b.shape[0] + offsety, offsetx:b.shape[1] + offsetx] = b  # 把图b拼接到tmp上去
            # result1 = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            # plt.imshow(result1)
            # plt.show()
            a = tmp  # 继续拼接左半部分的下一张图
        self.leftImage = tmp

    def rightshift(self):
        for each in self.right_list:  # 右半部分图
            H = self.matcher_obj.match(self.leftImage, each)
            #print("Homography :", H)
            txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            txyz = txyz / txyz[-1]
            dsize = (int(txyz[0]) + self.leftImage.shape[1], int(txyz[1]) + self.leftImage.shape[0])
            tmp = cv2.warpPerspective(each, H, dsize)
            result = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            plt.imshow(result)
            plt.show()
            dst = self.mix_and_match(self.leftImage, tmp)
            result1 = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
            plt.imshow(result1)
            plt.show()
            self.leftImage = dst
        return dst

    def mix_and_match(self, leftImage, warpedImage):
        i1y, i1x = leftImage.shape[:2]
        i2y, i2x = warpedImage.shape[:2]
        print(leftImage[-1, -1])
        black_l = np.where(leftImage == np.array([0, 0, 0]))  # 黑色像素坐标
        print(black_l)
        black_wi = np.where(warpedImage == np.array([0, 0, 0]))
        print(black_l[-1])
        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    # 如果同一位置在两幅图中都是黑色像素
                    if (np.array_equal(leftImage[j, i], np.array([0, 0, 0])) and np.array_equal(warpedImage[j, i],
                                                                                                np.array([0, 0, 0]))):
                        # print "BLACK"
                        # instead of just putting it with black,
                        # take average of all nearby values and avg it.
                        warpedImage[j, i] = [0, 0, 0]  # 黑色像素赋给它
                    else:
                        if (np.array_equal(warpedImage[j, i], [0, 0, 0])):  # 如果即将拼上去的图中，该像素是黑色
                            # print "PIXEL"
                            warpedImage[j, i] = leftImage[j, i]  # 原图像的像素赋给它
                        else:
                            if not np.array_equal(leftImage[j, i], [0, 0, 0]):  # 其他颜色像素
                                warpedImage[j, i] = leftImage[j, i]
                            # warpedImage[j, i] = [bl,gl,rl]	#原图像的像素赋给它
                except:
                    pass
        return warpedImage

class Stitch_1:
    def __init__(self, path, size):
        self.path = path
        self.size = size
        filenames = os.listdir(path)
        print('参与拼接的图片：')
        #print(filenames)
        self.images = []  # 参与拼接的图片
        for each in filenames:  # 把参与拼接的图片尺寸调到一致
            file_path = path + each
            print(file_path)
            input_image = cv2.imread(file_path)
            resize_shape = self.size
            input_image = cv2.resize(input_image, (resize_shape[0], resize_shape[1]))
            #input_image = self.cylindricalProjection(input_image)
            self.images.append(input_image)
        self.count = len(self.images)
        print('numbers of images: ', self.count)
        self.matcher_obj = matchers()


    def cylindricalProjection(self, img):
        rows = img.shape[0]
        cols = img.shape[1]

        # f = cols / (2 * math.tan(np.pi / 8))
        result = np.zeros_like(img)
        center_x = int(cols / 2)
        center_y = int(rows / 2)
        alpha = math.pi / 4
        f = cols / (2 * math.tan(alpha / 2))
        for y in range(rows):
            for x in range(cols):
                theta = math.atan((x - center_x) / f)
                point_x = int(f * math.tan((x - center_x) / f) + center_x)
                point_y = int((y - center_y) / math.cos(theta) + center_y)

                if point_x >= cols or point_x < 0 or point_y >= rows or point_y < 0:
                    pass
                else:
                    result[y, x, :] = img[point_y, point_x, :]
        return result

    def find_the_top(self, H, shape):
        # left top
        [w, h, tem] = shape
        v2 = [0, 0, 1]
        v1 = np.dot(H, v2)
        top = []
        top.append([v1[0] / v2[2], v1[1] / v2[2]])

        # left bottom
        v2[0] = 0
        v2[1] = w
        v2[2] = 1
        v1 = np.dot(H, v2)
        top.append([v1[0] / v2[2], v1[1] / v2[2]])

        # right top
        v2[0] = h
        v2[1] = 0
        v2[2] = 1
        v1 = np.dot(H, v2)
        top.append([v1[0] / v2[2], v1[1] / v2[2]])
        return top

    def two_in_one(self, imageA, imageB, begin_w, last_w):
        (hA, wA, tem) = imageA.shape
        (hB, wB, tem) = imageB.shape
        h = min(hA, hB)
        over = int(wA - begin_w - last_w)
        begin_w = int(begin_w)
        imageA[0:h, 0:begin_w] = imageB[0:h, 0:begin_w]
        for now_w in range(begin_w, wB):
            for now_h in range(0, h):
                alpha = (now_w * 1.0 - begin_w) / over * 1.0
                # 如果都是黑的
                if ((imageA[now_h][now_w][0] == 0) & (imageA[now_h][now_w][1] == 0) & (imageA[now_h][now_w][2] == 0)):
                    if ((imageB[now_h][now_w][0] == 0) & (imageB[now_h][now_w][1] == 0) & (imageB[now_h][now_w][2] == 0)):
                        imageA[now_h][now_w][0] = 0
                        imageA[now_h][now_w][1] = 0
                        imageA[now_h][now_w][2] = 0
                    # A黑B不黑，使用B的像素
                    else:
                        alpha = 0
                        imageA[now_h][now_w][0] = imageA[now_h][now_w][0] * alpha + imageB[now_h][now_w][0] * (1 - alpha)
                        imageA[now_h][now_w][1] = imageA[now_h][now_w][1] * alpha + imageB[now_h][now_w][1] * (1 - alpha)
                        imageA[now_h][now_w][2] = imageA[now_h][now_w][2] * alpha + imageB[now_h][now_w][2] * (1 - alpha)
                else:
                    # B黑A不黑，使用A的像素
                    if ((imageB[now_h][now_w][0] == 0) & (imageB[now_h][now_w][1] == 0) & (imageB[now_h][now_w][2] == 0)):
                        alpha = 1
                        imageA[now_h][now_w][0] = imageA[now_h][now_w][0] * alpha + imageB[now_h][now_w][0] * (1 - alpha)
                        imageA[now_h][now_w][1] = imageA[now_h][now_w][1] * alpha + imageB[now_h][now_w][1] * (1 - alpha)
                        imageA[now_h][now_w][2] = imageA[now_h][now_w][2] * alpha + imageB[now_h][now_w][2] * (1 - alpha)
                    # AB都不黑，像素加权
                    else:
                        imageA[now_h][now_w][0] = imageA[now_h][now_w][0] * alpha + imageB[now_h][now_w][0] * (1 - alpha)
                        imageA[now_h][now_w][1] = imageA[now_h][now_w][1] * alpha + imageB[now_h][now_w][1] * (1 - alpha)
                        imageA[now_h][now_w][2] = imageA[now_h][now_w][2] * alpha + imageB[now_h][now_w][2] * (1 - alpha)
        return imageA

    def shift(self):
        a = self.images[0]
        i = 15
        for b in self.images[1:]:
            H = self.matcher_obj.match(a, b)  # 特征点匹配
            top = self.find_the_top(H, b.shape)
            last_w = int(min(top[0][0], top[1][0]))
            # tmp：图b变换后的图像
            tmp = cv2.warpPerspective(b, H, (b.shape[1] + last_w, max(a.shape[0], b.shape[0])))
            cv2.imwrite('./output/{}_a.jpg'.format(i), a)
            cv2.imwrite('./output/{}_b.jpg'.format(i), b)
            cv2.imwrite('./output/{}_b_warped.jpg'.format(i), tmp)
            result = self.two_in_one(tmp, a, max(top[0][0], top[1][0]), last_w)
            cv2.imwrite('./output/{}_result.jpg'.format(i), result)
            i += 1
            width = 250
            a = cv2.resize(result, (b.shape[0], b.shape[1] - width))  # 为循环做准备
        return self.ExtractImage(result)

    def getBinaryImage(self, image):
        img = cv2.medianBlur(image, 3)  # 中值滤波
        b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        binary_image = b[1]  # 二值图--具有三通道
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        return binary_image

    def ExtractImage(self, image):
        # 去除边框，提取图像内容
        plt.figure(num='ExtractImage')

        image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        binary_image = self.getBinaryImage(image)
        cv2.imwrite("binary_image.png", binary_image)

        ret, thresh = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY)
        binary, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)  # 获取最大轮廓

        mask = np.zeros(thresh.shape, dtype="uint8")
        x, y, w, h = cv2.boundingRect(cnt)
        # 绘制最大外接矩形框（内部填充）
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        cv2.imwrite("mask.png", mask)

        minRect = mask.copy()
        sub = mask.copy()
        print(sub.shape[0] * sub.shape[1])
        # 连续腐蚀操作，直到sub中不再有前景像素
        while cv2.countNonZero(sub) > 0:
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)

        binary, cnts, hierarchy = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        image = image[y:y + h, x:x + w]
        return image

class Stitch_2:
    def __init__(self, path, size):
        self.path = path
        self.size = size
        filenames = os.listdir(path)
        print('参与拼接的图片：')
        #print(filenames)
        self.images = []  # 参与拼接的图片
        for each in filenames:  # 把参与拼接的图片尺寸调到一致
            file_path = path + each
            print(file_path)
            input_image = cv2.imread(file_path)
            resize_shape = self.size
            input_image = cv2.resize(input_image, (resize_shape[0], resize_shape[1]))
            #input_image = self.cylindricalProjection(input_image)
            self.images.append(input_image)
        self.count = len(self.images)
        print('numbers of images: ', self.count)
        self.matcher_obj = matchers()

    def auto_stitch(self):
        stitcher = cv2.createStitcher(cv2.Stitcher_PANORAMA)
        result = stitcher.stitch(self.images)
        return result[1]

if __name__ == '__main__':
    try:
        args = sys.argv[1]
    except:
        # 需要拼接图片所在的文件夹
        args = "./input/"
    finally:
        print("需要拼接图片所在的文件夹 : ", args)

    image_resize = [1000, 1000]  # 统一调整输入图片的大小

    # method 0 : 图像直接拼接
    # method 1 : 柱面投影 + 加权融合
    # method 2 : openCV自带的stitch类
    method = 0

    if method == 0:
        s = Stitch_0(args, image_resize)
        s.leftshift()
        leftimage = s.rightshift()

    if method == 1:
        s = Stitch_1(args, image_resize)
        leftimage = s.shift()

    if method == 2:
        s = Stitch_2(args, image_resize)
        leftimage = s.auto_stitch()

    print("image stitched")
    cv2.imwrite("./output/final_picture.jpg", leftimage)
    print("image written")
