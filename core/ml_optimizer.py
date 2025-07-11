"""
머신러닝 기반 전략 최적화 시스템
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class MLOptimizer:
    """머신러닝 기반 전략 최적화기"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.best_params = {}
        self.optimization_history = []
        
        # 모델 디렉토리 생성
        os.makedirs(model_dir, exist_ok=True)
        
        # 지원하는 모델 타입
        self.model_types = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'xgboost': xgb.XGBClassifier
        }
    
    def optimize_strategy_parameters(
        self,
        train_data: pd.DataFrame,
        strategy_name: str,
        param_ranges: Dict[str, Tuple],
        n_trials: int = 100,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        전략 파라미터 최적화
        
        Args:
            train_data: 훈련 데이터
            strategy_name: 전략 이름
            param_ranges: 파라미터 범위
            n_trials: 최적화 시도 횟수
            cv_folds: 교차검증 폴드 수
            
        Returns:
            Dict: 최적화 결과
        """
        try:
            logger.info(f"전략 {strategy_name} 파라미터 최적화 시작")
            
            def objective(trial):
                # 파라미터 샘플링
                params = {}
                for param_name, (min_val, max_val) in param_ranges.items():
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                    else:
                        params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                
                # 전략 시뮬레이션
                returns = self.simulate_strategy(train_data, strategy_name, params)
                
                # 목적 함수 (샤프 비율)
                if len(returns) > 0 and returns.std() > 0:
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                    return sharpe_ratio
                else:
                    return -999
            
            # Optuna 최적화
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            # 최적 파라미터 저장
            best_params = study.best_params
            best_value = study.best_value
            
            self.best_params[strategy_name] = best_params
            
            # 최적화 결과 저장
            result = {
                'strategy': strategy_name,
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': n_trials,
                'timestamp': datetime.now().isoformat()
            }
            
            self.optimization_history.append(result)
            self.save_optimization_results(strategy_name, result)
            
            logger.info(f"전략 {strategy_name} 최적화 완료: {best_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"전략 최적화 실패: {e}")
            return {}
    
    def train_ml_model(
        self,
        train_data: pd.DataFrame,
        model_type: str = 'xgboost',
        optimize_hyperparams: bool = True,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        머신러닝 모델 훈련
        
        Args:
            train_data: 훈련 데이터
            model_type: 모델 타입
            optimize_hyperparams: 하이퍼파라미터 최적화 여부
            n_trials: 최적화 시도 횟수
            
        Returns:
            Dict: 훈련 결과
        """
        try:
            logger.info(f"ML 모델 훈련 시작: {model_type}")
            
            # 피처와 타겟 분리
            feature_columns = [col for col in train_data.columns if col not in ['target', 'future_return']]
            X = train_data[feature_columns]
            y = train_data['target']
            
            # 데이터 전처리
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if optimize_hyperparams:
                # 하이퍼파라미터 최적화
                best_params = self.optimize_hyperparameters(
                    X_scaled, y, model_type, n_trials
                )
            else:
                # 기본 파라미터 사용
                best_params = self.get_default_params(model_type)
            
            # 모델 훈련
            model_class = self.model_types[model_type]
            model = model_class(**best_params)
            model.fit(X_scaled, y)
            
            # 모델 성능 평가
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
            
            # 모델 및 스케일러 저장
            self.models[model_type] = model
            self.scalers[model_type] = scaler
            
            self.save_model(model, scaler, model_type)
            
            result = {
                'model_type': model_type,
                'best_params': best_params,
                'cv_scores': cv_scores.tolist(),
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'feature_columns': feature_columns,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"ML 모델 훈련 완료: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            return result
            
        except Exception as e:
            logger.error(f"ML 모델 훈련 실패: {e}")
            return {}
    
    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        하이퍼파라미터 최적화
        
        Args:
            X: 피처 데이터
            y: 타겟 데이터
            model_type: 모델 타입
            n_trials: 최적화 시도 횟수
            
        Returns:
            Dict: 최적 하이퍼파라미터
        """
        try:
            def objective(trial):
                # 모델별 하이퍼파라미터 범위 설정
                if model_type == 'random_forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'random_state': 42
                    }
                elif model_type == 'gradient_boosting':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'random_state': 42
                    }
                elif model_type == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'random_state': 42
                    }
                
                # 모델 생성 및 교차검증
                model_class = self.model_types[model_type]
                model = model_class(**params)
                
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
                
                return cv_scores.mean()
            
            # Optuna 최적화
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            return study.best_params
            
        except Exception as e:
            logger.error(f"하이퍼파라미터 최적화 실패: {e}")
            return self.get_default_params(model_type)
    
    def get_default_params(self, model_type: str) -> Dict[str, Any]:
        """
        기본 하이퍼파라미터 반환
        
        Args:
            model_type: 모델 타입
            
        Returns:
            Dict: 기본 파라미터
        """
        default_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        }
        
        return default_params.get(model_type, {})
    
    def predict_signals(
        self,
        data: pd.DataFrame,
        model_type: str = 'xgboost'
    ) -> np.ndarray:
        """
        거래 신호 예측
        
        Args:
            data: 예측할 데이터
            model_type: 사용할 모델 타입
            
        Returns:
            np.ndarray: 예측 신호
        """
        try:
            if model_type not in self.models:
                logger.error(f"모델 {model_type}이 훈련되지 않음")
                return np.array([])
            
            model = self.models[model_type]
            scaler = self.scalers[model_type]
            
            # 피처 추출
            feature_columns = [col for col in data.columns if col not in ['target', 'future_return']]
            X = data[feature_columns]
            
            # 스케일링
            X_scaled = scaler.transform(X)
            
            # 예측
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"신호 예측 실패: {e}")
            return np.array([]), np.array([])
    
    def simulate_strategy(
        self,
        data: pd.DataFrame,
        strategy_name: str,
        params: Dict[str, Any]
    ) -> pd.Series:
        """
        전략 시뮬레이션
        
        Args:
            data: 시뮬레이션할 데이터
            strategy_name: 전략 이름
            params: 전략 파라미터
            
        Returns:
            pd.Series: 수익률
        """
        try:
            # 전략별 시뮬레이션 로직
            if strategy_name == 'triple_combo':
                return self.simulate_triple_combo(data, params)
            elif strategy_name == 'rsi_strategy':
                return self.simulate_rsi_strategy(data, params)
            elif strategy_name == 'ml_enhanced':
                return self.simulate_ml_enhanced(data, params)
            else:
                logger.warning(f"알 수 없는 전략: {strategy_name}")
                return pd.Series([])
                
        except Exception as e:
            logger.error(f"전략 시뮬레이션 실패: {e}")
            return pd.Series([])
    
    def simulate_triple_combo(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """트리플 콤보 전략 시뮬레이션"""
        try:
            rsi_period = params.get('rsi_period', 14)
            rsi_oversold = params.get('rsi_oversold', 30)
            rsi_overbought = params.get('rsi_overbought', 70)
            
            # RSI 계산
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 매매 신호 생성
            buy_signal = (rsi < rsi_oversold) & (data['MACD'] > data['MACD_Signal'])
            sell_signal = (rsi > rsi_overbought) & (data['MACD'] < data['MACD_Signal'])
            
            # 수익률 계산
            position = 0
            returns = []
            
            for i in range(len(data)):
                if buy_signal.iloc[i] and position == 0:
                    position = 1
                elif sell_signal.iloc[i] and position == 1:
                    position = 0
                
                if position == 1:
                    returns.append(data['Price_Change'].iloc[i])
                else:
                    returns.append(0)
            
            return pd.Series(returns)
            
        except Exception as e:
            logger.error(f"트리플 콤보 시뮬레이션 실패: {e}")
            return pd.Series([])
    
    def simulate_rsi_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """RSI 전략 시뮬레이션"""
        try:
            rsi_period = params.get('rsi_period', 14)
            rsi_oversold = params.get('rsi_oversold', 30)
            rsi_overbought = params.get('rsi_overbought', 70)
            
            # RSI 기반 매매 신호
            buy_signal = data['RSI'] < rsi_oversold
            sell_signal = data['RSI'] > rsi_overbought
            
            # 수익률 계산
            position = 0
            returns = []
            
            for i in range(len(data)):
                if buy_signal.iloc[i] and position == 0:
                    position = 1
                elif sell_signal.iloc[i] and position == 1:
                    position = 0
                
                if position == 1:
                    returns.append(data['Price_Change'].iloc[i])
                else:
                    returns.append(0)
            
            return pd.Series(returns)
            
        except Exception as e:
            logger.error(f"RSI 시뮬레이션 실패: {e}")
            return pd.Series([])
    
    def simulate_ml_enhanced(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """ML 강화 전략 시뮬레이션"""
        try:
            model_type = params.get('model_type', 'xgboost')
            confidence_threshold = params.get('confidence_threshold', 0.6)
            
            # ML 예측
            predictions, probabilities = self.predict_signals(data, model_type)
            
            if len(predictions) == 0:
                return pd.Series([])
            
            # 신뢰도 기반 매매 신호
            buy_signal = (predictions == 1) & (probabilities[:, 1] > confidence_threshold)
            sell_signal = (predictions == 0) & (probabilities[:, 0] > confidence_threshold)
            
            # 수익률 계산
            position = 0
            returns = []
            
            for i in range(len(data)):
                if buy_signal[i] and position == 0:
                    position = 1
                elif sell_signal[i] and position == 1:
                    position = 0
                
                if position == 1:
                    returns.append(data['Price_Change'].iloc[i])
                else:
                    returns.append(0)
            
            return pd.Series(returns)
            
        except Exception as e:
            logger.error(f"ML 강화 시뮬레이션 실패: {e}")
            return pd.Series([])
    
    def save_model(self, model: Any, scaler: Any, model_type: str):
        """모델 저장"""
        try:
            model_path = f"{self.model_dir}/{model_type}_model.pkl"
            scaler_path = f"{self.model_dir}/{model_type}_scaler.pkl"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"모델 저장 완료: {model_path}")
            
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
    
    def load_model(self, model_type: str) -> bool:
        """모델 로드"""
        try:
            model_path = f"{self.model_dir}/{model_type}_model.pkl"
            scaler_path = f"{self.model_dir}/{model_type}_scaler.pkl"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[model_type] = joblib.load(model_path)
                self.scalers[model_type] = joblib.load(scaler_path)
                logger.info(f"모델 로드 완료: {model_path}")
                return True
            else:
                logger.warning(f"모델 파일이 없음: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False
    
    def save_optimization_results(self, strategy_name: str, results: Dict[str, Any]):
        """최적화 결과 저장"""
        try:
            filename = f"{self.model_dir}/{strategy_name}_optimization_results.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"최적화 결과 저장: {filename}")
            
        except Exception as e:
            logger.error(f"최적화 결과 저장 실패: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """최적화 요약 정보 반환"""
        try:
            summary = {
                'total_optimizations': len(self.optimization_history),
                'strategies_optimized': list(self.best_params.keys()),
                'models_trained': list(self.models.keys()),
                'last_optimization': self.optimization_history[-1] if self.optimization_history else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"최적화 요약 생성 실패: {e}")
            return {}