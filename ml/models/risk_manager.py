"""
동적 ML 기반 리스크 관리 시스템
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from core.data_manager import DataManager
from core.backtest_engine import RealBacktestEngine

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """리스크 지표"""
    volatility: float
    var_95: float  # 95% VaR
    max_drawdown: float
    sharpe_ratio: float
    beta: float
    correlation_market: float
    liquidity_score: float
    momentum_score: float

@dataclass
class RiskParameters:
    """리스크 관리 파라미터"""
    max_position_size: float
    stop_loss_pct: float
    take_profit_pct: float
    max_leverage: float
    max_correlation: float
    min_liquidity: float
    risk_score: float
    recommended_allocation: float

class DynamicRiskManager:
    """동적 ML 기반 리스크 관리자"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.backtest_engine = None
        
        # ML 모델들
        self.volatility_model = None
        self.drawdown_model = None
        self.return_model = None
        self.risk_model = None
        
        # 스케일러들
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # 모델 경로
        self.model_dir = "ml/models/saved"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 기본 리스크 파라미터
        self.base_risk_params = {
            'max_position_size': 0.1,  # 포트폴리오의 10%
            'stop_loss_pct': 0.05,     # 5% 손절
            'take_profit_pct': 0.15,   # 15% 익절
            'max_leverage': 3.0,       # 최대 3배
            'max_correlation': 0.7,    # 최대 상관관계 70%
            'min_liquidity': 0.3       # 최소 유동성 점수 30%
        }
        
        # 시장 데이터 캐시
        self.market_data_cache = {}
        self.last_update = {}
        
    def get_backtest_engine(self):
        """백테스트 엔진 지연 초기화"""
        if self.backtest_engine is None:
            self.backtest_engine = RealBacktestEngine()
        return self.backtest_engine
    
    async def train_risk_models(self, lookback_days: int = 365) -> Dict[str, Any]:
        """ML 리스크 모델 훈련"""
        logger.info("🤖 ML 리스크 모델 훈련 시작")
        
        training_results = {
            'models_trained': [],
            'performance_metrics': {},
            'feature_importance': {},
            'training_errors': []
        }
        
        try:
            # 훈련 데이터 수집
            training_data = await self._collect_training_data(lookback_days)
            
            if training_data.empty:
                raise ValueError("훈련 데이터가 충분하지 않습니다")
            
            # 특성 엔지니어링
            features, targets = self._engineer_features(training_data)
            
            # 모델별 훈련
            models_config = {
                'volatility': {
                    'target': 'volatility',
                    'model_class': RandomForestRegressor,
                    'params': {'n_estimators': 100, 'random_state': 42}
                },
                'drawdown': {
                    'target': 'max_drawdown',
                    'model_class': GradientBoostingRegressor,
                    'params': {'n_estimators': 100, 'random_state': 42}
                },
                'return': {
                    'target': 'return_1d',
                    'model_class': RandomForestRegressor,
                    'params': {'n_estimators': 150, 'random_state': 42}
                },
                'risk': {
                    'target': 'risk_score',
                    'model_class': GradientBoostingRegressor,
                    'params': {'n_estimators': 120, 'random_state': 42}
                }
            }
            
            for model_name, config in models_config.items():
                try:
                    model, metrics = self._train_single_model(
                        features, targets[config['target']], 
                        config['model_class'], config['params']
                    )
                    
                    # 모델 저장
                    setattr(self, f"{model_name}_model", model)
                    self._save_model(model, f"{model_name}_model.pkl")
                    
                    training_results['models_trained'].append(model_name)
                    training_results['performance_metrics'][model_name] = metrics
                    
                    # 특성 중요도
                    if hasattr(model, 'feature_importances_'):
                        training_results['feature_importance'][model_name] = dict(
                            zip(features.columns, model.feature_importances_)
                        )
                    
                    logger.info(f"✅ {model_name} 모델 훈련 완료 (R² Score: {metrics['r2_score']:.3f})")
                    
                except Exception as e:
                    error_msg = f"{model_name} 모델 훈련 실패: {e}"
                    logger.error(error_msg)
                    training_results['training_errors'].append(error_msg)
            
            # 스케일러 저장
            self._save_scaler(self.feature_scaler, "feature_scaler.pkl")
            
            logger.info(f"🎯 ML 리스크 모델 훈련 완료: {len(training_results['models_trained'])}개 모델")
            
        except Exception as e:
            error_msg = f"ML 모델 훈련 중 오류: {e}"
            logger.error(error_msg)
            training_results['training_errors'].append(error_msg)
        
        return training_results
    
    async def _collect_training_data(self, lookback_days: int) -> pd.DataFrame:
        """훈련 데이터 수집"""
        logger.info(f"📊 최근 {lookback_days}일 백테스트 데이터 수집 중...")
        
        # 백테스트 엔진에서 주요 심볼들에 대한 백테스트 실행
        engine = self.get_backtest_engine()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        training_records = []
        
        # 주요 심볼들
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
        strategies = ['triple_combo', 'rsi_strategy']
        
        for symbol in symbols:
            for strategy in strategies:
                try:
                    # 30일 단위로 슬라이딩 윈도우 백테스트
                    current_date = start_date
                    while current_date < end_date - timedelta(days=30):
                        window_end = current_date + timedelta(days=30)
                        
                        # 백테스트 실행
                        result = await engine.run_backtest(
                            symbol=symbol,
                            strategy=strategy,
                            start_date=current_date.strftime('%Y-%m-%d'),
                            end_date=window_end.strftime('%Y-%m-%d'),
                            initial_capital=100000
                        )
                        
                        if result:
                            # 시장 데이터 로드
                            market_data = self.data_manager.load_data(symbol, '1h')
                            window_data = market_data[
                                (market_data.index >= current_date) & 
                                (market_data.index <= window_end)
                            ]
                            
                            if not window_data.empty:
                                # 특성 및 타겟 계산
                                record = self._calculate_training_features(
                                    window_data, result, symbol, strategy
                                )
                                training_records.append(record)
                        
                        current_date += timedelta(days=7)  # 7일씩 이동
                        
                except Exception as e:
                    logger.warning(f"데이터 수집 실패 ({symbol}, {strategy}): {e}")
        
        if training_records:
            training_df = pd.DataFrame(training_records)
            logger.info(f"✅ 훈련 데이터 수집 완료: {len(training_df)} 레코드")
            return training_df
        else:
            logger.warning("훈련 데이터가 수집되지 않았습니다")
            return pd.DataFrame()
    
    def _calculate_training_features(self, market_data: pd.DataFrame, backtest_result: Any, 
                                   symbol: str, strategy: str) -> Dict[str, float]:
        """훈련용 특성 계산"""
        close_prices = market_data['close']
        volumes = market_data['volume']
        
        # 가격 특성
        returns = close_prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(24)  # 일일 변동성
        
        # 기술적 지표
        rsi = self._calculate_rsi(close_prices, 14).iloc[-1]
        
        # 볼륨 특성
        avg_volume = volumes.mean()
        volume_trend = volumes.pct_change().mean()
        
        # 모멘텀 특성
        momentum_5d = (close_prices.iloc[-1] / close_prices.iloc[-5] - 1) if len(close_prices) >= 5 else 0
        momentum_20d = (close_prices.iloc[-1] / close_prices.iloc[-20] - 1) if len(close_prices) >= 20 else 0
        
        # 변동성 특성
        price_range = (market_data['high'] - market_data['low']) / market_data['close']
        avg_price_range = price_range.mean()
        
        # 백테스트 결과에서 타겟 변수들
        total_return = getattr(backtest_result, 'total_return', 0)
        max_drawdown = abs(getattr(backtest_result, 'max_drawdown', 0))
        sharpe_ratio = getattr(backtest_result, 'sharpe_ratio', 0)
        win_rate = getattr(backtest_result, 'win_rate', 0)
        
        # 리스크 점수 계산 (0-100)
        risk_score = self._calculate_risk_score(volatility, max_drawdown, sharpe_ratio)
        
        return {
            # 특성들
            'volatility_feature': volatility,
            'rsi': rsi,
            'avg_volume': avg_volume,
            'volume_trend': volume_trend,
            'momentum_5d': momentum_5d,
            'momentum_20d': momentum_20d,
            'avg_price_range': avg_price_range,
            'is_btc': 1 if 'BTC' in symbol else 0,
            'is_eth': 1 if 'ETH' in symbol else 0,
            'is_triple_combo': 1 if strategy == 'triple_combo' else 0,
            
            # 타겟들
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'return_1d': total_return / 30,  # 일일 수익률 근사치
            'risk_score': risk_score,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate
        }
    
    def _calculate_risk_score(self, volatility: float, max_drawdown: float, sharpe_ratio: float) -> float:
        """리스크 점수 계산 (0-100, 낮을수록 안전)"""
        # 각 요소를 0-100 스케일로 정규화
        vol_score = min(volatility * 1000, 100)  # 변동성
        dd_score = min(max_drawdown * 100, 100)  # 최대 손실
        sharpe_score = max(0, 50 - sharpe_ratio * 25)  # 샤프 비율 (높을수록 좋음)
        
        # 가중 평균
        risk_score = (vol_score * 0.4 + dd_score * 0.4 + sharpe_score * 0.2)
        return min(max(risk_score, 0), 100)
    
    def _engineer_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """특성 엔지니어링"""
        feature_columns = [
            'volatility_feature', 'rsi', 'avg_volume', 'volume_trend',
            'momentum_5d', 'momentum_20d', 'avg_price_range',
            'is_btc', 'is_eth', 'is_triple_combo'
        ]
        
        target_columns = [
            'volatility', 'max_drawdown', 'return_1d', 'risk_score'
        ]
        
        # 결측치 제거
        clean_data = data.dropna()
        
        features = clean_data[feature_columns]
        targets = clean_data[target_columns]
        
        # 특성 스케일링
        features_scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        return features_scaled, targets
    
    def _train_single_model(self, features: pd.DataFrame, target: pd.Series, 
                           model_class, params: Dict) -> Tuple[Any, Dict]:
        """단일 모델 훈련"""
        # 시계열 교차 검증
        tscv = TimeSeriesSplit(n_splits=5)
        
        cv_scores = []
        best_model = None
        best_score = float('-inf')
        
        for train_idx, val_idx in tscv.split(features):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
            
            # 모델 훈련
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            # 검증
            y_pred = model.predict(X_val)
            score = r2_score(y_val, y_pred)
            cv_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_model = model
        
        # 최종 모델 전체 데이터로 재훈련
        final_model = model_class(**params)
        final_model.fit(features, target)
        
        # 성능 지표
        y_pred_final = final_model.predict(features)
        
        metrics = {
            'r2_score': r2_score(target, y_pred_final),
            'mse': mean_squared_error(target, y_pred_final),
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores)
        }
        
        return final_model, metrics
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def assess_risk(self, market_data: pd.DataFrame, symbol: str) -> float:
        """실시간 리스크 평가"""
        try:
            # 모델이 없으면 기본 리스크 점수 반환
            if not self.risk_model:
                self.load_models()
            
            if not self.risk_model:
                return 50.0  # 중간 리스크
            
            # 특성 계산
            features = self._calculate_realtime_features(market_data, symbol)
            
            # 특성 스케일링
            features_scaled = self.feature_scaler.transform([features])
            
            # 리스크 점수 예측
            risk_score = self.risk_model.predict(features_scaled)[0]
            
            return max(0, min(100, risk_score))
            
        except Exception as e:
            logger.error(f"리스크 평가 실패: {e}")
            return 50.0
    
    def _calculate_realtime_features(self, market_data: pd.DataFrame, symbol: str) -> List[float]:
        """실시간 특성 계산"""
        close_prices = market_data['close']
        volumes = market_data['volume']
        
        # 가격 특성
        returns = close_prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(24) if len(returns) > 0 else 0
        
        # 기술적 지표
        rsi = self._calculate_rsi(close_prices, 14).iloc[-1] if len(close_prices) >= 14 else 50
        
        # 볼륨 특성
        avg_volume = volumes.mean() if len(volumes) > 0 else 0
        volume_trend = volumes.pct_change().mean() if len(volumes) > 1 else 0
        
        # 모멘텀 특성
        momentum_5d = (close_prices.iloc[-1] / close_prices.iloc[-5] - 1) if len(close_prices) >= 5 else 0
        momentum_20d = (close_prices.iloc[-1] / close_prices.iloc[-20] - 1) if len(close_prices) >= 20 else 0
        
        # 변동성 특성
        if 'high' in market_data.columns and 'low' in market_data.columns:
            price_range = (market_data['high'] - market_data['low']) / market_data['close']
            avg_price_range = price_range.mean()
        else:
            avg_price_range = 0
        
        # 심볼 특성
        is_btc = 1 if 'BTC' in symbol else 0
        is_eth = 1 if 'ETH' in symbol else 0
        is_triple_combo = 1  # 기본값
        
        return [
            volatility, rsi, avg_volume, volume_trend,
            momentum_5d, momentum_20d, avg_price_range,
            is_btc, is_eth, is_triple_combo
        ]
    
    async def calculate_optimal_risk_params(self, symbol: str, 
                                          strategy: str) -> RiskParameters:
        """최적 리스크 파라미터 계산"""
        try:
            # 시장 데이터 로드
            market_data = self.data_manager.load_data(symbol, '1h')
            
            if market_data.empty:
                return self._get_default_risk_params()
            
            # ML 예측
            risk_score = await self.assess_risk(market_data, symbol)
            
            # 변동성 예측
            predicted_volatility = self._predict_volatility(market_data, symbol)
            
            # 최대 손실 예측
            predicted_drawdown = self._predict_drawdown(market_data, symbol)
            
            # 리스크 기반 파라미터 조정
            risk_factor = risk_score / 100.0
            volatility_factor = min(predicted_volatility / 0.3, 2.0)  # 기준 변동성 30%
            
            # 동적 파라미터 계산
            max_position_size = self.base_risk_params['max_position_size'] / (1 + risk_factor)
            stop_loss_pct = self.base_risk_params['stop_loss_pct'] * (1 + volatility_factor * 0.5)
            take_profit_pct = self.base_risk_params['take_profit_pct'] * (1 + volatility_factor * 0.3)
            max_leverage = self.base_risk_params['max_leverage'] / (1 + risk_factor * 0.5)
            
            # 권장 할당 비중
            base_allocation = 0.2  # 기본 20%
            recommended_allocation = base_allocation * (1 - risk_factor * 0.5)
            
            return RiskParameters(
                max_position_size=round(max_position_size, 3),
                stop_loss_pct=round(stop_loss_pct, 3),
                take_profit_pct=round(take_profit_pct, 3),
                max_leverage=round(max_leverage, 1),
                max_correlation=self.base_risk_params['max_correlation'],
                min_liquidity=self.base_risk_params['min_liquidity'],
                risk_score=round(risk_score, 1),
                recommended_allocation=round(recommended_allocation, 3)
            )
            
        except Exception as e:
            logger.error(f"리스크 파라미터 계산 실패: {e}")
            return self._get_default_risk_params()
    
    def _predict_volatility(self, market_data: pd.DataFrame, symbol: str) -> float:
        """변동성 예측"""
        try:
            if self.volatility_model:
                features = self._calculate_realtime_features(market_data, symbol)
                features_scaled = self.feature_scaler.transform([features])
                return max(0, self.volatility_model.predict(features_scaled)[0])
            else:
                # 기본 계산
                returns = market_data['close'].pct_change().dropna()
                return returns.std() * np.sqrt(24) if len(returns) > 0 else 0.3
        except:
            return 0.3
    
    def _predict_drawdown(self, market_data: pd.DataFrame, symbol: str) -> float:
        """최대 손실 예측"""
        try:
            if self.drawdown_model:
                features = self._calculate_realtime_features(market_data, symbol)
                features_scaled = self.feature_scaler.transform([features])
                return max(0, self.drawdown_model.predict(features_scaled)[0])
            else:
                # 기본 계산 (최근 손실 패턴 분석)
                returns = market_data['close'].pct_change().dropna()
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                return abs(drawdown.min()) if len(drawdown) > 0 else 0.1
        except:
            return 0.1
    
    def _get_default_risk_params(self) -> RiskParameters:
        """기본 리스크 파라미터"""
        return RiskParameters(
            max_position_size=self.base_risk_params['max_position_size'],
            stop_loss_pct=self.base_risk_params['stop_loss_pct'],
            take_profit_pct=self.base_risk_params['take_profit_pct'],
            max_leverage=self.base_risk_params['max_leverage'],
            max_correlation=self.base_risk_params['max_correlation'],
            min_liquidity=self.base_risk_params['min_liquidity'],
            risk_score=50.0,
            recommended_allocation=0.2
        )
    
    def _save_model(self, model, filename: str):
        """모델 저장"""
        try:
            filepath = os.path.join(self.model_dir, filename)
            joblib.dump(model, filepath)
            logger.info(f"모델 저장: {filepath}")
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
    
    def _save_scaler(self, scaler, filename: str):
        """스케일러 저장"""
        try:
            filepath = os.path.join(self.model_dir, filename)
            joblib.dump(scaler, filepath)
            logger.info(f"스케일러 저장: {filepath}")
        except Exception as e:
            logger.error(f"스케일러 저장 실패: {e}")
    
    def load_models(self) -> bool:
        """저장된 모델들 로드"""
        try:
            model_files = {
                'volatility_model': 'volatility_model.pkl',
                'drawdown_model': 'drawdown_model.pkl',
                'return_model': 'return_model.pkl',
                'risk_model': 'risk_model.pkl'
            }
            
            loaded_count = 0
            
            for attr_name, filename in model_files.items():
                filepath = os.path.join(self.model_dir, filename)
                if os.path.exists(filepath):
                    try:
                        model = joblib.load(filepath)
                        setattr(self, attr_name, model)
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"모델 로드 실패 ({filename}): {e}")
            
            # 스케일러 로드
            scaler_path = os.path.join(self.model_dir, "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                try:
                    self.feature_scaler = joblib.load(scaler_path)
                    logger.info("특성 스케일러 로드 완료")
                except Exception as e:
                    logger.warning(f"스케일러 로드 실패: {e}")
            
            logger.info(f"✅ {loaded_count}개 ML 모델 로드 완료")
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류: {e}")
            return False