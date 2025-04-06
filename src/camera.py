import cv2
import time
import numpy as np


def create_mask(image, blur_size=15, threshold_value=8, morph_size=15):
    """
    为拼图块创建掩膜，排除边缘区域
    Args:
        image: 输入图像
        blur_size: 高斯模糊核大小
        threshold_value: 阈值化参数
        morph_size: 形态学操作核大小
    Returns:
        内部区域掩膜
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    edges = cv2.Canny(blurred, threshold_value, threshold_value * 3)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)

    if contours:
        # 找到最大轮廓（假设是拼图块的边界）
        max_contour = max(contours, key=cv2.contourArea)
        # 填充轮廓内部
        cv2.drawContours(mask, [max_contour], 0, 255, -1)
        # 腐蚀掩膜，去除边缘区域
        morph_kernel = np.ones((morph_size, morph_size), np.uint8)
        mask = cv2.erode(mask, morph_kernel, iterations=1)
    return mask


def open_camera(camera_index, crop_ratio, callback):
    cv2.destroyAllWindows()
    time.sleep(0.2)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: camera {camera_index} can not open")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2
    end_x = start_x + crop_width
    end_y = start_y + crop_height

    print(f"camera {camera_index} on")
    print(f"crop ratio: {int(crop_ratio * 100)}%")
    print("press ESC to exit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Unable to capture image")
                break
            center_area = frame[start_y:end_y, start_x:end_x].copy()
            mask = create_mask(center_area)
            if callback:
                callback(center_area, mask)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.2)
