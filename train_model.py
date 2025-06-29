from ml.models.price_prediction_model import PricePredictionModel
import pandas as pd

# 샘플 훈련 데이터 로드 (날짜를 인덱스로)
train_data = pd.read_csv('data/market_data/sample_train_data.csv', index_col=0)

# 모델 훈련
model = PricePredictionModel()
model.fit(train_data)
model.save_model('trained_model.pkl') 