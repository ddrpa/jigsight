import cv2
import numpy as np


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def match_with_sift(base_image, gray_base, piece, mask):
    sift = cv2.SIFT_create()
    gray_piece = preprocess(piece)

    kp1, des1 = sift.detectAndCompute(gray_piece, mask)
    kp2, des2 = sift.detectAndCompute(gray_base, None)

    if kp1 is None or len(kp1) < 10:
        print("特征点太少")
        return base_image, None

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"匹配点数量: {len(good)}")
    if len(good) <= 4:
        print("匹配点太少")
        return base_image, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        return base_image, None

    h, w = gray_piece.shape[:2]
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    # 绘制匹配区域
    matched = base_image.copy()
    cv2.polylines(matched, [np.int32(dst)], True, (0, 255, 0), 3)
    # 在匹配图上展示使用的关键点
    for i, m in enumerate(good):
        # 获取原图中的关键点坐标
        x2, y2 = kp2[m.trainIdx].pt
        cv2.circle(matched, (int(x2), int(y2)), 5, (255, 0, 255), -1)

    angle = -np.arctan2(M[0, 1], M[0, 0]) * 180 / np.pi
    return matched, angle
