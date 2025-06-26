import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(image_path, type_path):
    images = np.load(image_path, allow_pickle=True)
    types = np.load(type_path, allow_pickle=True)

    le = LabelEncoder() # 문자 라벨을 숫자로 바꿔주는 역할
    labels = le.fit_transform(types)

    return images, labels, le