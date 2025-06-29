import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import optuna

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

    def fit(self, df, target_col='close', horizon=1, tune=False):
        df_feat = make_features(df)
        X = df_feat.drop([target_col], axis=1).values
        y = df_feat[target_col].shift(-horizon).dropna().values
        X = X[:len(y)]  # y와 길이 맞추기

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
                for train_idx, val_idx in tscv.split(X):
                    model.fit(X[train_idx], y[train_idx])
                    preds = model.predict(X[val_idx])
                    score = mean_squared_error(y[val_idx], preds, squared=False)
                    scores.append(score)
                return np.mean(scores)
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20)
            self.best_params['rf'] = study.best_params
            self.models['rf'] = RandomForestRegressor(**study.best_params)

        # 각 모델 학습
        for name, model in self.models.items():
            model.fit(X, y)

    def predict(self, df):
        df_feat = make_features(df)
        X = df_feat.drop(['close'], axis=1).values
        preds = []
        for name, model in self.models.items():
            preds.append(model.predict(X))
        # 앙상블(평균)
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