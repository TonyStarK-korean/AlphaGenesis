from ml.models.price_prediction_model import PricePredictionModel
import pandas as pd

# (예시) 훈련 데이터 로드 - 실제 사용 데이터 경로로 수정하세요!
train_data = pd.read_csv('data/your_train_data.csv')

# 모델 훈련
model = PricePredictionModel()
model.fit(train_data)
model.save_model('trained_model.pkl') 