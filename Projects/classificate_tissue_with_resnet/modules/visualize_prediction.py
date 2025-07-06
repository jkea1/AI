import os
import torch
import random
import matplotlib.pyplot as plt

def visualize_predictions(model, val_dataset, label_encoder, device, num_samples=10, save_path=None):
    model.eval()
    
    indices = random.sample(range(len(val_dataset)), num_samples)
    correct = 0

    plt.figure(figsize=(15, 8))

    for i, idx in enumerate(indices):
        img, true_label = val_dataset[idx]

        # 배치 차원 추가 & device로 이동
        img_input = img.unsqueeze(0).to(device)

        # 예측
        with torch.no_grad():
            outputs = model(img_input)
            _, predicted = torch.max(outputs, 1)

        # 정답 여부 확인
        if predicted.item() == true_label:
            correct += 1

        # 라벨 변환
        true_class = label_encoder.inverse_transform([true_label])[0]
        predicted_class = label_encoder.inverse_transform([predicted.cpu().item()])[0]

        # 이미지 디노멀라이즈 (정규화 해제)
        img_np = img.permute(1, 2, 0).cpu().numpy()  # [C, H, W] → [H, W, C]
        img_np = img_np * 0.5 + 0.5  # [-1,1] → [0,1]

        # 시각화
        plt.subplot(2, 5, i + 1)
        plt.imshow(img_np)
        plt.title(f"GT: {true_class}\nPred: {predicted_class}", fontsize=10)
        plt.axis("off")

    plt.suptitle(f"{correct}/{num_samples} Correct", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Predictions saved to {save_path}")

    plt.show()