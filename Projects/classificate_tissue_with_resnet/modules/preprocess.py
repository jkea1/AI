import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import os

def generate_processed_dataset(
    image_file_path="./data/raw/panNuke_images.npy",
    type_file_path="./data/raw/panNuke_types.npy",
    save_dir="./data/processed",
    image_size=224,
    test_size=0.2,
    random_state=42
):
    # 1. Load raw data
    images = np.load(image_file_path)  # (N, H, W, 3)
    raw_labels = np.load(type_file_path)  # (N,)

    # 2. Label encoding
    le = LabelEncoder() # string label -> integer label
    labels = le.fit_transform(raw_labels)

    # 3. Resize + ToTensor
    transform = transforms.Compose([
      transforms.Resize((image_size, image_size)), # 이미 PIL 이미지
      transforms.ToTensor() # PIL (N, H, W, C=3) -> Tensor (N, C=3, H, W)
    ])

    print("Transforming and saving images...")

    processed_images = []

    for img in tqdm(images):
      pil_img = Image.fromarray(img.astype("uint8")) # numpy -> PIL
      tensor_img = transform(pil_img) # PIL -> Tensor
      processed_images.append(tensor_img.numpy())  # numpy로 저장

    processed_images = np.stack(processed_images)

    # 4. Split
    # 80% train / 20% validation
    # stratify=labels => Breast, Colon, … 모든 클래스가 train/val에 비슷한 비율로 포함됨
    # random_state => train/validation 으로 나눌 때 마다 랜덤으로 나눠지도록 한다.
    train_idx, val_idx = train_test_split(
      np.arange(len(labels)),
      test_size=test_size,
      stratify=labels,
      random_state=random_state
    )

    # 5. Save
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "resized_images.npy"), processed_images)
    np.save(os.path.join(save_dir, "labels.npy"), labels)
    np.save(os.path.join(save_dir, "train_indices.npy"), train_idx)
    np.save(os.path.join(save_dir, "val_indices.npy"), val_idx)

    with open(os.path.join(save_dir, "classes.txt"), "w") as f:
      for c in le.classes_:
        f.write(c + "\n")

    print("✅ 전처리 및 저장 완료.")