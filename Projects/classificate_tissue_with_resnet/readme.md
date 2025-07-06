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

  ### 이미지 데이터 변환 흐름 요약

  | 단계                        | Shape            | dtype      | 값 범위   | 설명                             |
  |-----------------------------|------------------|------------|-----------|----------------------------------|
  | Raw 이미지 (`npy`)       | (224, 224, 3)    | `uint8`    | 0 ~ 255   | 일반 RGB 이미지 (H, W, C)        |
  | ToTensor 변환 후         | (3, 224, 224)    | `float32`  | 0.0 ~ 1.0 | Tensor 형태로 변환 (C, H, W)     |
  | 저장된 `.npy`            | (3, 224, 224)    | `float32`  | 0.0 ~ 1.0 | numpy 배열로 저장된 전처리 이미지 |
  | 불러와서 Tensor로 변환   | (3, 224, 224)    | `float32`  | 0.0 ~ 1.0 | `torch.from_numpy()` 사용         |
  | Normalize 적용 후        | (3, 224, 224)    | `float32`  | -1.0 ~ 1.0| 최종 모델 입력용 정규화 완료     |


### 모델 평가 지표

| 지표              | 정의                                               | 공식                                                                 | 해석/의미                                                           |
|-------------------|----------------------------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------|
| **Loss**          | 예측과 정답 간의 오차를 수치화한 값               | CrossEntropyLoss 사용                                                | 낮을수록 예측이 정답에 가까움                                      |
| **Accuracy**      | 전체 중 맞춘 정답 비율                             | $$\frac{\text{Correct}}{\text{Total}}$$           | 전체적인 성능을 빠르게 파악할 수 있음                              |
| **Precision**     | Positive로 예측한 것 중 실제로 Positive인 비율    | $$\frac{TP}{TP + FP}$$                             | False Positive를 줄이고 싶은 상황에서 중요                         |
| **Recall**        | 실제 Positive 중에서 모델이 맞춘 비율             | $$\frac{TP}{TP + FN}$$                             | 놓치면 안 되는 항목(FN)을 줄이고 싶은 상황에서 중요                |
| **F1-score**      | Precision과 Recall의 조화 평균                     | $$2 \cdot \frac{P \cdot R}{P + R}$$         | Precision과 Recall의 균형이 필요할 때 적합                         |
| **Confusion Matrix** | 예측 라벨과 실제 라벨의 분포 행렬            | 행: 실제 라벨, 열: 예측 라벨                                        | 클래스별로 잘못 예측한 부분을 시각적으로 확인 가능                 |

---

#### 추가 설명: `average="macro"`

- Precision, Recall, F1-score에서 사용됨
- 모든 클래스를 **동일한 가중치**로 평균 계산
- 클래스 불균형이 있는 경우에도 **균형 잡힌 평가 가능**

#### 추가 설명: `TP (True Positive)`, `FP (False Positive)`, `FN (False Negative)`, `TN (True Negative)`
- ex1. 예시 상황: 암 진단 모델
  | 이름                  | 무슨 뜻일까?                                      | 예시                                               |
  |---------------------|--------------------------------------------------|----------------------------------------------------|
  | ✅ True Positive (TP) | 진짜 암이고, 모델도 암이라고 맞췄어                | 암 환자를 **정확히** 암이라고 예측                  |
  | ✅ True Negative (TN) | 진짜 건강하고, 모델도 건강하다고 했어              | 건강한 사람을 **정확히** 건강하다고 예측             |
  | ❌ False Positive (FP) | 건강한 사람인데, 모델이 암이라고 잘못 예측했어     | **괜히 무서운 진단**을 받는 경우 (스트레스, 추가 검사 필요) |
  | ❌ False Negative (FN) | 암 환자인데, 모델이 건강하다고 잘못 예측했어       | **암을 놓친 상황** (가장 위험해요!!)                 |

- ex2. 예시 상황: 조직 분류 (Tissue Classification) - “Colon” 클래스 기준으로
  - 다중 분류(Multi-class classification)에서는 각 클래스(e.g. Colon)에 대해서 **이진 분류**처럼 생각해서 TP/FP/FN/TN를 구한다.

    | 모델의 예측 | 실제 정답 | 분류     | 해석                                 |
    |-------------|------------|----------|--------------------------------------|
    | Colon       | Colon      | ✅ TP     | Colon을 잘 맞췄어!                   |
    | Colon       | Liver      | ❌ FP     | Colon이라고 했는데 아니었어          |
    | Breast      | Colon      | ❌ FN     | Colon인데 모델이 놓쳤어              |
    | Breast      | Liver      | ✅ TN     | Colon이 아닌 것도 잘 구별했어         |