import os
import cv2
import numpy as np
import torch

def make_folder(folder_path):
    # 폴더가 존재하는지 확인하고, 없으면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def visualize_and_save_grad_cam(input_image, grad_cam, original_save_path, grad_cam_save_path, datasets, alpha=0.5):

    """
    Grad-CAM 이미지를 원본 이미지와 알파 처리하여 오버레이하고 저장하는 함수.
    input_image: 원본 이미지 (torch.Tensor, shape: (1, C, H, W))
    grad_cam: Grad-CAM 결과 (numpy 배열, shape: (H, W))
    original_save_path: 원본 이미지를 저장할 파일 경로 (str)
    grad_cam_save_path: Grad-CAM 이미지를 저장할 파일 경로 (str)
    datasets : datasets
    alpha: 원본 이미지와 Grad-CAM 오버레이 비율 (기본값: 0.5)
    """

    # Grad-CAM 결과를 [0, 1] 범위로 정규화
    grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam))

    # 원본 이미지를 (H, W, C)로 변환하고 [0, 255] 범위로 스케일링
    input_image = input_image.cpu().squeeze()
    input_image = input_image.permute(1, 2, 0)  # (channels, height, width) -> (height, width, channels)

    # Normalize의 반대로 [0, 1] 범위로 복원
    if datasets == "mini_imagenet":
        input_image = input_image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor(
            [0.485, 0.456, 0.406])
    elif datasets == "tiered_imagenet":
        input_image = input_image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor(
            [0.485, 0.456, 0.406])
    elif datasets == "CIFAR_FS":
        input_image = input_image * torch.tensor([0.2675, 0.2565, 0.2761]) + torch.tensor([0.5071, 0.4847, 0.4408])
    elif datasets == "CUB":
        input_image = input_image * torch.tensor([1 / 255.0, 1 / 255.0, 1 / 255.0]) + torch.tensor(
            [104 / 255.0, 117 / 255.0, 128 / 255.0])

    input_image = torch.clamp(input_image, 0, 1)  # 클리핑을 통해 값을 [0, 1]로 제한
    input_image = input_image.numpy()
    input_image = (input_image * 255).astype(np.uint8)

    # Grad-CAM을 컬러맵(Jet)을 적용하여 컬러 이미지로 변환
    grad_cam_colored = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)

    # 원본 이미지와 Grad-CAM을 오버레이
    overlay_image = cv2.addWeighted(input_image, alpha, grad_cam_colored, 1 - alpha, 0)

    # 원본 이미지 저장
    cv2.imwrite(original_save_path, input_image)

    # Grad-CAM 이미지 저장
    cv2.imwrite(grad_cam_save_path, overlay_image)

    # 시각화
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(input_image)
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.title("Grad-CAM")
#     plt.imshow(input_image, alpha=0.5)
#     plt.imshow(grad_cam, cmap='jet', alpha=0.5)
#     plt.axis("off")
#     plt.show()