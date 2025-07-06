### 이미지 변환 과정
- `numpy` (raw data) -> `PIL.Image` (for preprocessing) -> `PyTorch Tensor` (model input format)

1. numpy 이미지: np.ndarray
- .npy나 .jpg를 열면 처음에는 대부분 numpy array로 되어 있다.
- 모양: (H, W, 3)
-> ex: (224, 224, 3)

  ```
  type(img) # numpy.ndarray
  img.shape # (224, 224, 3)
  ```

2. PIL 이미지: PIL.Image.Image
- PyTorch의 transforms.Resize, transforms.RandomCrop 같은 변환 함수들은 PIL 이미지에 최적화돼 있다. 그래서 transform 하기 전에 numpy -> PIL로 바꿔줘야 한다.

  ```
  from PIL import Image

  img = Image.fromarray(img_numpy.astype("unit8")) # numpy -> PIL
  ```
- 이미 값이 0~255라도, 자료형이 uint8이 아니면 PIL.Image.fromarray()는 작동을 안 하거나 색이 깨지기 때문에, 자료형을 정확히 astype("uint8")로 바꿔주는 게 필수이다.

3. PyTorch Tensor
- 모델에 들어가려면 이미지가 Tensor 형식이어야 한다.
- `transforms.ToTensor()`는 PIL 이미지를 -> Tensor로 바꾸고, 동시에 정규화까지 해준다.

  | 항목       | numpy / PIL         | tensor (PyTorch 형식) |
  |------------|----------------------|------------------------|
  | **모양**   | (H, W, C)            | (C, H, W)              |
  | **값 범위** | 0 ~ 255 (정수형)     | 0.0 ~ 1.0 (실수형)     |

  ```
  from torchvision import transforms

  img_tensor = transforms.ToTensor()(img_pil)  # PIL → Tensor
  ```

  ### ✅ 이미지 데이터 변환 흐름 요약

  | 단계                        | Shape            | dtype      | 값 범위   | 설명                             |
  |-----------------------------|------------------|------------|-----------|----------------------------------|
  | 🔹 Raw 이미지 (`npy`)       | (224, 224, 3)    | `uint8`    | 0 ~ 255   | 일반 RGB 이미지 (H, W, C)        |
  | 🔄 ToTensor 변환 후         | (3, 224, 224)    | `float32`  | 0.0 ~ 1.0 | Tensor 형태로 변환 (C, H, W)     |
  | 💾 저장된 `.npy`            | (3, 224, 224)    | `float32`  | 0.0 ~ 1.0 | numpy 배열로 저장된 전처리 이미지 |
  | 📤 불러와서 Tensor로 변환   | (3, 224, 224)    | `float32`  | 0.0 ~ 1.0 | `torch.from_numpy()` 사용         |
  | 🎯 Normalize 적용 후        | (3, 224, 224)    | `float32`  | -1.0 ~ 1.0| 최종 모델 입력용 정규화 완료     |
