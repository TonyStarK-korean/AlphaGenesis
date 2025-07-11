<<<<<<< HEAD
import numpy as np
import pandas as pd
import warnings
import time
import os
import logging

# Try to import optional dependencies
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn이 설치되지 않았습니다. 기본 모델을 사용합니다.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# 경고 메시지 필터링
warnings.filterwarnings("ignore", message="X does not have valid feature names, but.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# scikit-learn 경고 완전 제거
os.environ['PYTHONWARNINGS'] = 'ignore'

# LightGBM 경고 완전 제거
logging.getLogger('lightgbm').setLevel(logging.CRITICAL)
os.environ['LIGHTGBM_VERBOSE'] = '0'

# XGBoost 경고 제거
logging.getLogger('xgboost').setLevel(logging.CRITICAL)

def make_features(df):
    # 실전에서 많이 쓰는 피처 예시
    df = df.copy()
    
    # 필수 컬럼 확인
    required_cols = ['close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[ML 모델] 필수 컬럼 누락: {missing_cols}")
        return df
    
    # 기본 피처 생성 (NaN 허용)
    df['return_1'] = df['close'].pct_change()
    df['ma_5'] = df['close'].rolling(5, min_periods=1).mean()
    df['ma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['volatility'] = df['close'].rolling(10, min_periods=1).std()
    df['volume_ma_5'] = df['volume'].rolling(5, min_periods=1).mean()
    
    # RSI 계산
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14, min_periods=1).mean()
    ma_down = down.rolling(14, min_periods=1).mean()
    df['rsi_14'] = 100 - (100 / (1 + ma_up / (ma_down + 1e-9)))
    
    # 기존 멀티타임프레임 지표가 있다면 유지
    # 없으면 기본값으로 채움
    if 'rsi_14_1h' not in df.columns:
        df['rsi_14_1h'] = df['rsi_14']
    if 'rsi_14_4h' not in df.columns:
        df['rsi_14_4h'] = df['rsi_14']
    if 'rsi_14_5m' not in df.columns:
        df['rsi_14_5m'] = df['rsi_14']
    
    if 'ema_20_1h' not in df.columns:
        df['ema_20_1h'] = df['ma_20']
    if 'ema_50_1h' not in df.columns:
        df['ema_50_1h'] = df['ma_20']
    if 'ema_120_1h' not in df.columns:
        df['ema_120_1h'] = df['ma_20']
    
    if 'macd_1h' not in df.columns:
        df['macd_1h'] = 0
    if 'macd_signal_1h' not in df.columns:
        df['macd_signal_1h'] = 0
    
    if 'vwap_1h' not in df.columns:
        df['vwap_1h'] = df['close']
    if 'bb_upper_1h' not in df.columns:
        df['bb_upper_1h'] = df['close'] * 1.02
    if 'bb_lower_1h' not in df.columns:
        df['bb_lower_1h'] = df['close'] * 0.98
    
    if 'stoch_k_5m' not in df.columns:
        df['stoch_k_5m'] = 50
    if 'stoch_d_5m' not in df.columns:
        df['stoch_d_5m'] = 50
    
    # NaN 값을 적절히 처리 (앞쪽 NaN은 0으로, 뒤쪽 NaN은 forward fill)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['close', 'volume']:  # 원본 데이터는 건드리지 않음
            continue
        try:
            # pandas 최신 버전 호환성을 위한 안전한 방법
            df[col] = df[col].ffill().fillna(0)
        except:
            # fallback: 더 안전한 방법
            df[col] = df[col].fillna(0)
    
    # 최종적으로 완전히 NaN인 행만 제거
    df = df.dropna(subset=['close', 'volume'])
    
    return df

class SimpleDummyModel:
    """간단한 더미 모델 (sklearn이 없을 때 사용)"""
    def __init__(self):
        self.last_price = None
        
    def fit(self, X, y):
        # 단순히 마지막 가격을 저장
        if len(y) > 0:
            self.last_price = y[-1]
        return self
        
    def predict(self, X):
        # 입력 크기만큼 마지막 가격 반복
        if self.last_price is None:
            return np.zeros(len(X))
        return np.full(len(X), self.last_price)

class PricePredictionModel:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.models = {}
        self.best_params = {}
        self.cv_report = {}
        self.feature_names = None
        self.sklearn_available = SKLEARN_AVAILABLE
        self.is_fitted = False  # 추가: 모델 훈련 상태 추적

    def save_model(self, path):
        if SKLEARN_AVAILABLE:
            joblib.dump({'model': self, 'cv_report': self.cv_report}, path)
            print(f"[모델저장] 모델이 {path}에 저장되었습니다. (CV리포트 포함)")
        else:
            # 단순히 pickle 사용
            import pickle
            with open(path, 'wb') as f:
                pickle.dump({'model': self, 'cv_report': self.cv_report}, f)
            print(f"[모델저장] 모델이 {path}에 저장되었습니다. (더미 모델)")

    @staticmethod
    def load_model(path):
        print(f"[모델불러오기] {path}에서 모델을 불러옵니다.")
        try:
            if SKLEARN_AVAILABLE:
                obj = joblib.load(path)
            else:
                import pickle
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
            
            if isinstance(obj, dict) and 'model' in obj:
                model = obj['model']
                model.cv_report = obj.get('cv_report', {})
                return model
            return obj
        except Exception as e:
            print(f"⚠️ 모델 로드 실패: {e}. 새 모델을 생성합니다.")
            return PricePredictionModel()

    def fit(self, df, target_col='close', horizon=1, tune=False):
        # sklearn이 없는 경우 더미 모델 사용
        if not SKLEARN_AVAILABLE:
            print("[ML 모델] sklearn이 없으므로 더미 모델을 사용합니다.")
            self.models = {'dummy': SimpleDummyModel()}
            if len(df) > 0:
                self.models['dummy'].fit(None, df[target_col].values)
            self.is_fitted = True  # 더미 모델 훈련 완료 표시
            return True
            
        # 최소 데이터 요구사항 체크
        if len(df) < 50:  # 최소 데이터 요구사항을 낮춤
            print(f"[ML 모델] 데이터 부족: {len(df)}개 (최소 50개 필요)")
            return False
            
        df_feat = make_features(df)
        
        if len(df_feat) < 20:  # 피처 생성 후 최소 요구사항을 낮춤
            print(f"[ML 모델] 피처 생성 후 데이터 부족: {len(df_feat)}개 (최소 20개 필요)")
            return False
        
        # 사용 가능한 피처 확인
        available_features = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [target_col, 'symbol', 'timestamp']
        feature_cols = [col for col in available_features if col not in exclude_cols]
        
        if len(feature_cols) < 5:  # 최소 피처 수 체크
            print(f"[ML 모델] 사용 가능한 피처 부족: {len(feature_cols)}개 (최소 5개 필요)")
            return False
            
        X = df_feat[feature_cols]
        y = df_feat[target_col].shift(-horizon).dropna().values
        X = X[:len(y)]  # y와 길이 맞추기
        
        if len(X) < 15:  # 최종 훈련 데이터 최소 요구사항을 낮춤
            print(f"[ML 모델] 최종 훈련 데이터 부족: {len(X)}개 (최소 15개 필요)")
            return False
            
        self.feature_names = X.columns.tolist()
        X = X.values
        
        # NaN 값 처리
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        
        if np.isnan(y).any():
            valid_indices = ~np.isnan(y)
            X = X[valid_indices]
            y = y[valid_indices]
            
        if len(X) < 10:  # 최종 검증
            print(f"[ML 모델] 최종 훈련 데이터 부족: {len(X)}개 (최소 10개 필요)")
            return False

        # 앙상블 모델 정의 (사용 가능한 것만)
        self.models = {}
        if SKLEARN_AVAILABLE:
            self.models['rf'] = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=0)
            self.models['ridge'] = Ridge(alpha=1.0)
        if XGBOOST_AVAILABLE:
            self.models['xgb'] = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, verbosity=0)
        if LIGHTGBM_AVAILABLE:
            self.models['lgb'] = lgb.LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
        
        if not self.models:
            print("[ML 모델] 사용 가능한 ML 라이브러리가 없습니다. 더미 모델을 사용합니다.")
            self.models = {'dummy': SimpleDummyModel()}
            self.models['dummy'].fit(None, y)
            self.is_fitted = True  # 더미 모델 훈련 완료 표시
            return True

        # 하이퍼파라미터 튜닝 (Optuna) - 데이터가 충분한 경우에만
        if tune and len(X) >= 30 and OPTUNA_AVAILABLE and SKLEARN_AVAILABLE:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 8)
                }
                model = RandomForestRegressor(**params, verbose=0)
                tscv = TimeSeriesSplit(n_splits=min(3, len(X)//10))  # fold 수 축소
                scores = []
                for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    if len(train_idx) < 5 or len(val_idx) < 3:  # 최소 요구사항 축소
                        continue
                    model.fit(X[train_idx], y[train_idx])
                    preds = model.predict(X[val_idx])
                    score = np.sqrt(mean_squared_error(y[val_idx], preds))
                    scores.append(score)
                if not scores:
                    return float('inf')
                mean_score = np.mean(scores)
                return mean_score
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=min(10, len(X)//3))  # trial 수 축소
            if study.best_params:
                self.best_params['rf'] = study.best_params
                self.models['rf'] = RandomForestRegressor(**study.best_params, verbose=0)
                print(f"[ML 모델] 최적화 완료: {self.best_params['rf']}")

        # 각 모델 학습 및 성능 리포트
        if SKLEARN_AVAILABLE:
            tscv = TimeSeriesSplit(n_splits=min(3, len(X)//10))  # fold 수 축소
        else:
            # 더미 모델은 이미 훈련되었으므로 성공 리턴
            self.is_fitted = True  # 더미 모델 훈련 완료 표시
            return True
        for name, model in self.models.items():
            fold_rmse, fold_mae, fold_r2 = [], [], []
            for train_idx, val_idx in tscv.split(X):
                if len(train_idx) < 5 or len(val_idx) < 3:  # 최소 요구사항 축소
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
                print(f"[ML 모델] {name} 모델 훈련 완료 - RMSE: {np.mean(fold_rmse):.2f}, R²: {np.mean(fold_r2):.3f}")
            else:
                print(f"[ML 모델] {name} 모델 훈련 실패")
                self.is_fitted = False  # 훈련 실패 표시
                return False
        
        print(f"[ML 모델] 모든 모델 훈련 완료 - 데이터: {len(X)}개, 피처: {len(self.feature_names)}개")
        self.is_fitted = True  # 훈련 완료 표시
        return True

    def predict(self, df):
        try:
            # sklearn이 없고 더미 모델을 사용하는 경우
            if not SKLEARN_AVAILABLE and 'dummy' in self.models:
                return self.models['dummy'].predict(np.zeros((len(df), 1)))
                
            # 모델 훈련 상태 체크
            if not hasattr(self, 'models') or not self.models:
                return np.zeros(len(df) if hasattr(df, '__len__') else 1)
            
            # feature_names 체크
            if not hasattr(self, 'feature_names') or self.feature_names is None:
                return np.zeros(len(df) if hasattr(df, '__len__') else 1)
            
            target_col = 'close'
            df_feat = make_features(df)
            
            # 필요한 컬럼이 없는 경우 처리
            missing_features = [col for col in self.feature_names if col not in df_feat.columns]
            if missing_features:
                return np.zeros(len(df_feat))
            
            # 학습 때와 동일한 feature만 사용
            X = df_feat[self.feature_names].values
            
            # NaN 값 처리
            if np.isnan(X).any():
                X = np.nan_to_num(X, nan=0.0)
            
            preds = []
            trained_models = 0
            
            for name, model in self.models.items():
                try:
                    # 모델이 훈련되었는지 체크
                    if hasattr(model, 'predict') and hasattr(model, 'fit'):
                        # scikit-learn 모델의 경우
                        if hasattr(model, 'estimators_') or hasattr(model, 'coef_') or hasattr(model, 'intercept_'):
                            pred = model.predict(X)
                            preds.append(pred)
                            trained_models += 1
                        # XGBoost 모델의 경우
                        elif hasattr(model, 'booster') and model.booster is not None:
                            pred = model.predict(X)
                            preds.append(pred)
                            trained_models += 1
                        # LightGBM 모델의 경우
                        elif hasattr(model, 'booster_') and model.booster_ is not None:
                            pred = model.predict(X)
                            preds.append(pred)
                            trained_models += 1
                        else:
                            preds.append(np.zeros(len(X)))
                    else:
                        preds.append(np.zeros(len(X)))
                        
                except Exception as e:
                    preds.append(np.zeros(len(X)))
            
            if trained_models == 0:
                return np.zeros(len(df_feat))
            
            if not preds:
                return np.zeros(len(df_feat))
            
            # 훈련된 모델들의 예측만 평균
            valid_preds = [pred for pred in preds if not np.all(pred == 0)]
            if valid_preds:
                return np.mean(valid_preds, axis=0)
            else:
                return np.zeros(len(df_feat))
            
        except Exception as e:
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
=======
import numpy as np
import pandas as pd
import warnings
import time
import os
import logging

# Try to import optional dependencies
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn이 설치되지 않았습니다. 기본 모델을 사용합니다.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# 경고 메시지 필터링
warnings.filterwarnings("ignore", message="X does not have valid feature names, but.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# scikit-learn 경고 완전 제거
os.environ['PYTHONWARNINGS'] = 'ignore'

# LightGBM 경고 완전 제거
logging.getLogger('lightgbm').setLevel(logging.CRITICAL)
os.environ['LIGHTGBM_VERBOSE'] = '0'

# XGBoost 경고 제거
logging.getLogger('xgboost').setLevel(logging.CRITICAL)

def make_features(df):
    # 실전에서 많이 쓰는 피처 예시
    df = df.copy()
    
    # 필수 컬럼 확인
    required_cols = ['close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[ML 모델] 필수 컬럼 누락: {missing_cols}")
        return df
    
    # 기본 피처 생성 (NaN 허용)
    df['return_1'] = df['close'].pct_change()
    df['ma_5'] = df['close'].rolling(5, min_periods=1).mean()
    df['ma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['volatility'] = df['close'].rolling(10, min_periods=1).std()
    df['volume_ma_5'] = df['volume'].rolling(5, min_periods=1).mean()
    
    # RSI 계산
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14, min_periods=1).mean()
    ma_down = down.rolling(14, min_periods=1).mean()
    df['rsi_14'] = 100 - (100 / (1 + ma_up / (ma_down + 1e-9)))
    
    # 기존 멀티타임프레임 지표가 있다면 유지
    # 없으면 기본값으로 채움
    if 'rsi_14_1h' not in df.columns:
        df['rsi_14_1h'] = df['rsi_14']
    if 'rsi_14_4h' not in df.columns:
        df['rsi_14_4h'] = df['rsi_14']
    if 'rsi_14_5m' not in df.columns:
        df['rsi_14_5m'] = df['rsi_14']
    
    if 'ema_20_1h' not in df.columns:
        df['ema_20_1h'] = df['ma_20']
    if 'ema_50_1h' not in df.columns:
        df['ema_50_1h'] = df['ma_20']
    if 'ema_120_1h' not in df.columns:
        df['ema_120_1h'] = df['ma_20']
    
    if 'macd_1h' not in df.columns:
        df['macd_1h'] = 0
    if 'macd_signal_1h' not in df.columns:
        df['macd_signal_1h'] = 0
    
    if 'vwap_1h' not in df.columns:
        df['vwap_1h'] = df['close']
    if 'bb_upper_1h' not in df.columns:
        df['bb_upper_1h'] = df['close'] * 1.02
    if 'bb_lower_1h' not in df.columns:
        df['bb_lower_1h'] = df['close'] * 0.98
    
    if 'stoch_k_5m' not in df.columns:
        df['stoch_k_5m'] = 50
    if 'stoch_d_5m' not in df.columns:
        df['stoch_d_5m'] = 50
    
    # NaN 값을 적절히 처리 (앞쪽 NaN은 0으로, 뒤쪽 NaN은 forward fill)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['close', 'volume']:  # 원본 데이터는 건드리지 않음
            continue
        try:
            # pandas 최신 버전 호환성을 위한 안전한 방법
            df[col] = df[col].ffill().fillna(0)
        except:
            # fallback: 더 안전한 방법
            df[col] = df[col].fillna(0)
    
    # 최종적으로 완전히 NaN인 행만 제거
    df = df.dropna(subset=['close', 'volume'])
    
    return df

class SimpleDummyModel:
    """간단한 더미 모델 (sklearn이 없을 때 사용)"""
    def __init__(self):
        self.last_price = None
        
    def fit(self, X, y):
        # 단순히 마지막 가격을 저장
        if len(y) > 0:
            self.last_price = y[-1]
        return self
        
    def predict(self, X):
        # 입력 크기만큼 마지막 가격 반복
        if self.last_price is None:
            return np.zeros(len(X))
        return np.full(len(X), self.last_price)

class PricePredictionModel:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.models = {}
        self.best_params = {}
        self.cv_report = {}
        self.feature_names = None
        self.sklearn_available = SKLEARN_AVAILABLE

    def save_model(self, path):
        if SKLEARN_AVAILABLE:
            joblib.dump({'model': self, 'cv_report': self.cv_report}, path)
            print(f"[모델저장] 모델이 {path}에 저장되었습니다. (CV리포트 포함)")
        else:
            # 단순히 pickle 사용
            import pickle
            with open(path, 'wb') as f:
                pickle.dump({'model': self, 'cv_report': self.cv_report}, f)
            print(f"[모델저장] 모델이 {path}에 저장되었습니다. (더미 모델)")

    @staticmethod
    def load_model(path):
        print(f"[모델불러오기] {path}에서 모델을 불러옵니다.")
        try:
            if SKLEARN_AVAILABLE:
                obj = joblib.load(path)
            else:
                import pickle
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
            
            if isinstance(obj, dict) and 'model' in obj:
                model = obj['model']
                model.cv_report = obj.get('cv_report', {})
                return model
            return obj
        except Exception as e:
            print(f"⚠️ 모델 로드 실패: {e}. 새 모델을 생성합니다.")
            return PricePredictionModel()

    def fit(self, df, target_col='close', horizon=1, tune=False):
        # sklearn이 없는 경우 더미 모델 사용
        if not SKLEARN_AVAILABLE:
            print("[ML 모델] sklearn이 없으므로 더미 모델을 사용합니다.")
            self.models = {'dummy': SimpleDummyModel()}
            if len(df) > 0:
                self.models['dummy'].fit(None, df[target_col].values)
            return True
            
        # 최소 데이터 요구사항 체크
        if len(df) < 50:  # 최소 데이터 요구사항을 낮춤
            print(f"[ML 모델] 데이터 부족: {len(df)}개 (최소 50개 필요)")
            return False
            
        df_feat = make_features(df)
        
        if len(df_feat) < 20:  # 피처 생성 후 최소 요구사항을 낮춤
            print(f"[ML 모델] 피처 생성 후 데이터 부족: {len(df_feat)}개 (최소 20개 필요)")
            return False
        
        # 사용 가능한 피처 확인
        available_features = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [target_col, 'symbol', 'timestamp']
        feature_cols = [col for col in available_features if col not in exclude_cols]
        
        if len(feature_cols) < 5:  # 최소 피처 수 체크
            print(f"[ML 모델] 사용 가능한 피처 부족: {len(feature_cols)}개 (최소 5개 필요)")
            return False
            
        X = df_feat[feature_cols]
        y = df_feat[target_col].shift(-horizon).dropna().values
        X = X[:len(y)]  # y와 길이 맞추기
        
        if len(X) < 15:  # 최종 훈련 데이터 최소 요구사항을 낮춤
            print(f"[ML 모델] 최종 훈련 데이터 부족: {len(X)}개 (최소 15개 필요)")
            return False
            
        self.feature_names = X.columns.tolist()
        X = X.values
        
        # NaN 값 처리
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        
        if np.isnan(y).any():
            valid_indices = ~np.isnan(y)
            X = X[valid_indices]
            y = y[valid_indices]
            
        if len(X) < 10:  # 최종 검증
            print(f"[ML 모델] 최종 훈련 데이터 부족: {len(X)}개 (최소 10개 필요)")
            return False

        # 앙상블 모델 정의 (사용 가능한 것만)
        self.models = {}
        if SKLEARN_AVAILABLE:
            self.models['rf'] = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=0)
            self.models['ridge'] = Ridge(alpha=1.0)
        if XGBOOST_AVAILABLE:
            self.models['xgb'] = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, verbosity=0)
        if LIGHTGBM_AVAILABLE:
            self.models['lgb'] = lgb.LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
        
        if not self.models:
            print("[ML 모델] 사용 가능한 ML 라이브러리가 없습니다. 더미 모델을 사용합니다.")
            self.models = {'dummy': SimpleDummyModel()}
            self.models['dummy'].fit(None, y)
            return True

        # 하이퍼파라미터 튜닝 (Optuna) - 데이터가 충분한 경우에만
        if tune and len(X) >= 30 and OPTUNA_AVAILABLE and SKLEARN_AVAILABLE:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 8)
                }
                model = RandomForestRegressor(**params, verbose=0)
                tscv = TimeSeriesSplit(n_splits=min(3, len(X)//10))  # fold 수 축소
                scores = []
                for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    if len(train_idx) < 5 or len(val_idx) < 3:  # 최소 요구사항 축소
                        continue
                    model.fit(X[train_idx], y[train_idx])
                    preds = model.predict(X[val_idx])
                    score = np.sqrt(mean_squared_error(y[val_idx], preds))
                    scores.append(score)
                if not scores:
                    return float('inf')
                mean_score = np.mean(scores)
                return mean_score
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=min(10, len(X)//3))  # trial 수 축소
            if study.best_params:
                self.best_params['rf'] = study.best_params
                self.models['rf'] = RandomForestRegressor(**study.best_params, verbose=0)
                print(f"[ML 모델] 최적화 완료: {self.best_params['rf']}")

        # 각 모델 학습 및 성능 리포트
        if SKLEARN_AVAILABLE:
            tscv = TimeSeriesSplit(n_splits=min(3, len(X)//10))  # fold 수 축소
        else:
            # 더미 모델은 이미 훈련되었으므로 성공 리턴
            return True
        for name, model in self.models.items():
            fold_rmse, fold_mae, fold_r2 = [], [], []
            for train_idx, val_idx in tscv.split(X):
                if len(train_idx) < 5 or len(val_idx) < 3:  # 최소 요구사항 축소
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
                print(f"[ML 모델] {name} 모델 훈련 완료 - RMSE: {np.mean(fold_rmse):.2f}, R²: {np.mean(fold_r2):.3f}")
            else:
                print(f"[ML 모델] {name} 모델 훈련 실패")
                return False
        
        print(f"[ML 모델] 모든 모델 훈련 완료 - 데이터: {len(X)}개, 피처: {len(self.feature_names)}개")
        return True

    def predict(self, df):
        try:
            # sklearn이 없고 더미 모델을 사용하는 경우
            if not SKLEARN_AVAILABLE and 'dummy' in self.models:
                return self.models['dummy'].predict(np.zeros((len(df), 1)))
                
            # 모델 훈련 상태 체크
            if not hasattr(self, 'models') or not self.models:
                return np.zeros(len(df) if hasattr(df, '__len__') else 1)
            
            # feature_names 체크
            if not hasattr(self, 'feature_names') or self.feature_names is None:
                return np.zeros(len(df) if hasattr(df, '__len__') else 1)
            
            target_col = 'close'
            df_feat = make_features(df)
            
            # 필요한 컬럼이 없는 경우 처리
            missing_features = [col for col in self.feature_names if col not in df_feat.columns]
            if missing_features:
                return np.zeros(len(df_feat))
            
            # 학습 때와 동일한 feature만 사용
            X = df_feat[self.feature_names].values
            
            # NaN 값 처리
            if np.isnan(X).any():
                X = np.nan_to_num(X, nan=0.0)
            
            preds = []
            trained_models = 0
            
            for name, model in self.models.items():
                try:
                    # 모델이 훈련되었는지 체크
                    if hasattr(model, 'predict') and hasattr(model, 'fit'):
                        # scikit-learn 모델의 경우
                        if hasattr(model, 'estimators_') or hasattr(model, 'coef_') or hasattr(model, 'intercept_'):
                            pred = model.predict(X)
                            preds.append(pred)
                            trained_models += 1
                        # XGBoost 모델의 경우
                        elif hasattr(model, 'booster') and model.booster is not None:
                            pred = model.predict(X)
                            preds.append(pred)
                            trained_models += 1
                        # LightGBM 모델의 경우
                        elif hasattr(model, 'booster_') and model.booster_ is not None:
                            pred = model.predict(X)
                            preds.append(pred)
                            trained_models += 1
                        else:
                            preds.append(np.zeros(len(X)))
                    else:
                        preds.append(np.zeros(len(X)))
                        
                except Exception as e:
                    preds.append(np.zeros(len(X)))
            
            if trained_models == 0:
                return np.zeros(len(df_feat))
            
            if not preds:
                return np.zeros(len(df_feat))
            
            # 훈련된 모델들의 예측만 평균
            valid_preds = [pred for pred in preds if not np.all(pred == 0)]
            if valid_preds:
                return np.mean(valid_preds, axis=0)
            else:
                return np.zeros(len(df_feat))
            
        except Exception as e:
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
>>>>>>> febb08c8d864666b98f9587b4eb4ce3a55eed692
    print("샤프지수:", result['sharpe'])