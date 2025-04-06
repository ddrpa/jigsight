import time

import cv2
import numpy as np

from camera import open_camera
from detector.sift import match_with_sift

global base_image, gray_base, masked_preview
global last_processing_time
global last_result


def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    return rotated


def display(piece, mask, matched_result, angle):
    h_match, w_match = matched_result.shape[:2]
    left_ratio = 0.7
    total_width = int(w_match / left_ratio)
    left_width = int(total_width * left_ratio)
    right_width = total_width - left_width

    panel_height = h_match // 3
    total_height = h_match

    result = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    left_height = total_height
    left_img = cv2.resize(matched_result, (left_width, left_height))
    result[0:left_height, 0:left_width] = left_img

    right_x = left_width
    piece_resized = cv2.resize(piece, (right_width, panel_height))
    result[0:panel_height, right_x:total_width] = piece_resized

    # 摄像头预览
    margin_ratio = 0.1  # 边距比例
    margin_h = int(panel_height * margin_ratio)
    margin_w = int(right_width * margin_ratio)
    cv2.rectangle(
        result,
        (right_x + margin_w, margin_h),
        (total_width - margin_w, panel_height - margin_h),
        (0, 255, 0), 2
    )

    # 掩膜处理后图像
    masked_piece = create_masked_piece(mask, piece)
    masked_resized = cv2.resize(masked_piece, (right_width, panel_height))
    result[panel_height:panel_height*2, right_x:total_width] = masked_resized

    # 匹配成功时显示旋转图像
    if angle is not None:
        rotated_piece = rotate_image(piece, angle)
        rotated_resized = cv2.resize(rotated_piece, (right_width, panel_height))
        result[panel_height*2:total_height, right_x:total_width] = rotated_resized
        angle_text = f"rotated: {angle:.1f}°"
        cv2.putText(
            result,
            angle_text,
            (right_x + 10, panel_height*2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 0), 1
        )
    else:
        # 匹配失败时右下区域保持空白
        pass
    return result


def callback_process_frame(piece, mask):
    matched_result, angle = match_with_sift(base_image, gray_base, piece, mask)
    result_display = display(piece, mask, matched_result, angle)
    cv2.imshow("jigsaw", result_display)


def create_masked_piece(mask, piece):
    # 创建掩膜效果预览
    mask_3channel = cv2.merge([mask, mask, mask])
    mask_3channel = mask_3channel.astype(np.float32) / 255
    darkened = (piece * 0.3).astype(np.uint8)
    masked_area = (piece * mask_3channel).astype(np.uint8)
    # 使用掩膜合并
    masked_piece = darkened.copy()
    masked_piece = np.where(mask_3channel > 0, masked_area, masked_piece)
    # 添加掩膜边界线
    mask_outline = cv2.dilate(mask, np.ones((3, 3), np.uint8)) - mask
    mask_outline_idx = mask_outline > 0
    if np.any(mask_outline_idx):
        masked_piece[mask_outline_idx] = [0, 0, 255]
    return masked_piece


def main():
    # 初始化全局变量
    global base_image, gray_base
    camera_index = 0
    crop_ratio = 0.4
    base_image = cv2.imread("../base.jpg")
    if base_image is None:
        print("错误：无法加载原始拼图图像")
        return
    # 计算原图的灰度图像
    gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("jigsaw", cv2.WINDOW_NORMAL)
    open_camera(camera_index, crop_ratio, callback_process_frame)


if __name__ == "__main__":
    main()
