import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import optuna
import time
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
import os
import logging

# 경고 메시지 필터링
warnings.filterwarnings("ignore", message="X does not have valid feature names, but.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# scikit-learn 경고 완전 제거
os.environ['PYTHONWARNINGS'] = 'ignore'

# LightGBM 경고 완전 제거
logging.getLogger('lightgbm').setLevel(logging.CRITICAL)

def make_features(df):
    # 실전에서 많이 쓰는 피처 예시
    df = df.copy()
    df['return_1'] = df['close'].pct_change()
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['volatility'] = df['close'].rolling(10).std()
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    # RSI 계산
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + ma_up / (ma_down + 1e-9)))
    df = df.dropna()
    return df

class PricePredictionModel:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.models = {}
        self.best_params = {}
        self.cv_report = {}
        self.feature_names = None

    def save_model(self, path):
        joblib.dump({'model': self, 'cv_report': self.cv_report}, path)
        print(f"[모델저장] 모델이 {path}에 저장되었습니다. (CV리포트 포함)")

    @staticmethod
    def load_model(path):
        print(f"[모델불러오기] {path}에서 모델을 불러옵니다.")
        obj = joblib.load(path)
        if isinstance(obj, dict) and 'model' in obj:
            model = obj['model']
            model.cv_report = obj.get('cv_report', {})
            return model
        return obj

    def fit(self, df, target_col='close', horizon=1, tune=False):
        # 최소 데이터 요구사항 체크
        if len(df) < 100:  # 최소 100개 데이터 포인트 필요
            print(f"[ML 모델] 데이터 부족: {len(df)}개 (최소 100개 필요)")
            return False
            
        df_feat = make_features(df)
        if len(df_feat) < 50:  # 피처 생성 후 최소 50개 필요
            print(f"[ML 모델] 피처 생성 후 데이터 부족: {len(df_feat)}개 (최소 50개 필요)")
            return False
            
        X = df_feat.drop([target_col, 'symbol', 'timestamp'], axis=1, errors='ignore')
        y = df_feat[target_col].shift(-horizon).dropna().values
        X = X[:len(y)]  # y와 길이 맞추기
        
        if len(X) < 30:  # 최종 훈련 데이터 최소 30개 필요
            print(f"[ML 모델] 최종 훈련 데이터 부족: {len(X)}개 (최소 30개 필요)")
            return False
            
        self.feature_names = X.columns.tolist()
        X = X.values

        # 앙상블 모델 정의
        self.models = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=200, max_depth=8, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=200, max_depth=8, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }

        # 하이퍼파라미터 튜닝 (Optuna)
        if tune and len(X) >= 50:  # 튜닝은 더 많은 데이터가 필요
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                    'max_depth': trial.suggest_int('max_depth', 3, 12)
                }
                model = RandomForestRegressor(**params)
                tscv = TimeSeriesSplit(n_splits=min(self.n_splits, len(X)//10))
                scores = []
                for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    if len(train_idx) < 10 or len(val_idx) < 5:  # 최소 훈련/검증 데이터 체크
                        continue
                    model.fit(X[train_idx], y[train_idx])
                    preds = model.predict(X[val_idx])
                    score = np.sqrt(mean_squared_error(y[val_idx], preds))
                    scores.append(score)
                if not scores:
                    return float('inf')
                mean_score = np.mean(scores)
                print(f"[Optuna 튜닝] trial={trial.number}, RMSE={mean_score:.4f}")
                return mean_score
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=min(20, len(X)//5))
            if study.best_params:
                self.best_params['rf'] = study.best_params
                self.models['rf'] = RandomForestRegressor(**study.best_params)
                print(f"[Optuna 최적화] best_params: {self.best_params['rf']}")

        # 각 모델 학습 및 성능 리포트
        tscv = TimeSeriesSplit(n_splits=min(self.n_splits, len(X)//10))
        for name, model in self.models.items():
            fold_rmse, fold_mae, fold_r2 = [], [], []
            for train_idx, val_idx in tscv.split(X):
                if len(train_idx) < 10 or len(val_idx) < 5:  # 최소 훈련/검증 데이터 체크
                    continue
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[val_idx])
                fold_rmse.append(np.sqrt(mean_squared_error(y[val_idx], preds)))
                fold_mae.append(mean_absolute_error(y[val_idx], preds))
                fold_r2.append(r2_score(y[val_idx], preds))
            
            if fold_rmse:  # 성공적으로 훈련된 경우만 리포트 저장
                self.cv_report[name] = {
                    'RMSE': np.mean(fold_rmse),
                    'MAE': np.mean(fold_mae),
                    'R2': np.mean(fold_r2)
                }
                print(f"[CV리포트] {name}: RMSE={np.mean(fold_rmse):.4f}, MAE={np.mean(fold_mae):.4f}, R2={np.mean(fold_r2):.4f}")
            else:
                print(f"[ML 모델] {name} 모델 훈련 실패 - 데이터 부족")
                return False
        
        print(f"[ML 모델] 모든 모델 훈련 완료. 훈련 데이터: {len(X)}개")
        return True

    def predict(self, df):
        try:
            target_col = 'close'
            df_feat = make_features(df)
            
            # feature_names가 None인 경우 처리
            if self.feature_names is None:
                print("[ML 모델] feature_names가 None입니다. 모델을 다시 훈련해야 합니다.")
                return np.zeros(len(df_feat))
            
            # 필요한 컬럼이 없는 경우 처리
            missing_features = [col for col in self.feature_names if col not in df_feat.columns]
            if missing_features:
                print(f"[ML 모델] 누락된 피처: {missing_features}")
                return np.zeros(len(df_feat))
            
            # 학습 때와 동일한 feature만 사용
            X = df_feat[self.feature_names].values
            
            # NaN 값 처리
            if np.isnan(X).any():
                print("[ML 모델] NaN 값이 발견되어 0으로 대체합니다.")
                X = np.nan_to_num(X, nan=0.0)
            
            preds = []
            for name, model in self.models.items():
                try:
                    pred = model.predict(X)
                    preds.append(pred)
                except Exception as e:
                    print(f"[ML 모델] {name} 모델 예측 실패: {e}")
                    # 예측 실패 시 0으로 대체
                    preds.append(np.zeros(len(X)))
            
            if not preds:
                print("[ML 모델] 모든 모델 예측 실패")
                return np.zeros(len(df_feat))
            
            return np.mean(preds, axis=0)
            
        except Exception as e:
            print(f"[ML 모델] 예측 중 오류 발생: {e}")
            # 오류 발생 시 0으로 반환
            return np.zeros(len(df) if hasattr(df, '__len__') else 1)

    def backtest(self, df, initial_capital=1000000, fee=0.0005, horizon=1):
        df_feat = make_features(df)
        preds = self.predict(df)
        df_feat = df_feat.iloc[:len(preds)]
        df_feat['pred'] = preds
        df_feat['signal'] = np.where(df_feat['pred'].shift(1) < df_feat['pred'], 1, -1)
        df_feat['ret'] = df_feat['close'].pct_change().shift(-horizon)
        df_feat['strategy_ret'] = df_feat['ret'] * df_feat['signal'] - fee
        df_feat['cum_ret'] = (1 + df_feat['strategy_ret']).cumprod() * initial_capital
        # 리스크 관리: 최대 드로우다운, 샤프지수 등
        max_dd = (df_feat['cum_ret'].cummax() - df_feat['cum_ret']).max()
        sharpe = df_feat['strategy_ret'].mean() / (df_feat['strategy_ret'].std() + 1e-9) * np.sqrt(252)
        return {
            'final_capital': df_feat['cum_ret'].iloc[-1],
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'history': df_feat
        }

# 사용 예시
if __name__ == "__main__":
    # 예시 데이터 (실전에서는 실제 OHLCV 데이터 사용)
    dates = pd.date_range('2022-01-01', periods=300)
    df = pd.DataFrame({
        'close': np.cumsum(np.random.randn(300)) + 100,
        'volume': np.random.randint(100, 1000, 300)
    }, index=dates)
    model = PricePredictionModel()
    model.fit(df, tune=True)
    result = model.backtest(df)
    print("최종 자본:", result['final_capital'])
    print("최대 드로우다운:", result['max_drawdown'])
    print("샤프지수:", result['sharpe'])