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

# 경고 메시지 필터링
warnings.filterwarnings("ignore", message="X does not have valid feature names, but.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# scikit-learn 경고 완전 제거
os.environ['PYTHONWARNINGS'] = 'ignore'

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
        print("[DEBUG] fit 진입, feature_names 존재 여부:", hasattr(self, 'feature_names'))
        df_feat = make_features(df)
        X = df_feat.drop([target_col, 'symbol', 'timestamp'], axis=1, errors='ignore')
        y = df_feat[target_col].shift(-horizon).dropna().values
        X = X[:len(y)]  # y와 길이 맞추기
        self.feature_names = X.columns.tolist()
        print("[DEBUG] fit 종료, feature_names:", self.feature_names)
        X = X.values

        # 앙상블 모델 정의
        self.models = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=200, max_depth=8, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=200, max_depth=8, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }

        # 하이퍼파라미터 튜닝 (Optuna)
        if tune:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                    'max_depth': trial.suggest_int('max_depth', 3, 12)
                }
                model = RandomForestRegressor(**params)
                tscv = TimeSeriesSplit(n_splits=self.n_splits)
                scores = []
                for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    model.fit(X[train_idx], y[train_idx])
                    preds = model.predict(X[val_idx])
                    score = np.sqrt(mean_squared_error(y[val_idx], preds))
                    scores.append(score)
                mean_score = np.mean(scores)
                print(f"[Optuna 튜닝] trial={trial.number}, RMSE={mean_score:.4f}")
                return mean_score
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20)
            self.best_params['rf'] = study.best_params
            self.models['rf'] = RandomForestRegressor(**study.best_params)
            print(f"[Optuna 최적화] best_params: {self.best_params['rf']}")

        # 각 모델 학습 및 성능 리포트
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for name, model in self.models.items():
            fold_rmse, fold_mae, fold_r2 = [], [], []
            for train_idx, val_idx in tscv.split(X):
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[val_idx])
                fold_rmse.append(np.sqrt(mean_squared_error(y[val_idx], preds)))
                fold_mae.append(mean_absolute_error(y[val_idx], preds))
                fold_r2.append(r2_score(y[val_idx], preds))
            self.cv_report[name] = {
                'RMSE': np.mean(fold_rmse),
                'MAE': np.mean(fold_mae),
                'R2': np.mean(fold_r2)
            }
            print(f"[CV리포트] {name}: RMSE={np.mean(fold_rmse):.4f}, MAE={np.mean(fold_mae):.4f}, R2={np.mean(fold_r2):.4f}")

    def predict(self, df):
        print("[DEBUG] predict 진입, feature_names:", getattr(self, 'feature_names', None))
        target_col = 'close'
        df_feat = make_features(df)
        # 학습 때와 동일한 feature만 사용
        X = df_feat[self.feature_names].values
        preds = []
        for name, model in self.models.items():
            preds.append(model.predict(X))
        return np.mean(preds, axis=0)

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