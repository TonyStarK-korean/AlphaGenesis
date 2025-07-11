"""
통합 설정 파일
모든 모듈의 설정을 중앙 집중식으로 관리
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import json
from pathlib import Path

class UnifiedConfig:
    """통합 설정 클래스"""
    
    def __init__(self):
        # 프로젝트 경로
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        self.results_dir = self.project_root / 'results'
        self.logs_dir = self.project_root / 'logs'
        
        # 기본 설정
        self._init_basic_settings()
        
        # 거래 설정
        self._init_trading_settings()
        
        # 백테스트 설정
        self._init_backtest_settings()
        
        # ML 모델 설정
        self._init_ml_settings()
        
        # 위험 관리 설정
        self._init_risk_management_settings()
        
        # 알림 설정
        self._init_notification_settings()
        
        # 데이터 다운로드 설정
        self._init_data_settings()
        
        # 대시보드 설정
        self._init_dashboard_settings()
        
    def _init_basic_settings(self):
        """기본 설정 초기화"""
        self.basic = {
            'project_name': 'AlphaGenesis',
            'version': '3.0.0',
            'environment': 'development',  # development, production
            'timezone': 'Asia/Seoul',
            'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
            'enable_debug': True,
            'max_workers': 4,  # 멀티스레딩 워커 수
        }
        
    def _init_trading_settings(self):
        """거래 설정 초기화"""
        self.trading = {
            # 기본 자본 설정
            'initial_capital': 10_000_000,  # 1000만원
            'target_capital': 100_000_000,  # 1억원
            'minimum_capital': 1_000_000,   # 100만원 (최소 자본)
            
            # 포지션 관리
            'max_position_size': 0.1,       # 최대 포지션 크기 (10%)
            'min_position_size': 0.01,      # 최소 포지션 크기 (1%)
            'position_scale_factor': 1.5,   # 포지션 스케일링 팩터
            
            # 손익 관리
            'default_stop_loss': 0.02,      # 기본 손절 (2%)
            'default_take_profit': 0.05,    # 기본 익절 (5%)
            'max_daily_loss': 0.05,         # 일일 최대 손실 (5%)
            'max_drawdown': 0.15,           # 최대 낙폭 (15%)
            
            # 레버리지 설정
            'max_leverage': 5.0,            # 최대 레버리지
            'default_leverage': 2.0,        # 기본 레버리지
            'conservative_leverage': 1.5,   # 보수적 레버리지
            'aggressive_leverage': 3.0,     # 공격적 레버리지
            
            # 수수료 설정
            'maker_fee': 0.0005,            # 메이커 수수료 (0.05%)
            'taker_fee': 0.001,             # 테이커 수수료 (0.1%)
            'withdrawal_fee': 0.0001,       # 출금 수수료 (0.01%)
            
            # 슬리피지 설정
            'slippage_tolerance': 0.0002,   # 슬리피지 허용치 (0.02%)
        }
        
    def _init_backtest_settings(self):
        """백테스트 설정 초기화"""
        self.backtest = {
            # 기본 백테스트 기간
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2024, 12, 31),
            
            # 백테스트 모드
            'mode': 'comprehensive',  # simple, comprehensive, walk_forward
            'timeframe': '1h',        # 1m, 5m, 15m, 1h, 4h, 1d
            'warm_up_period': 200,    # 워밍업 기간
            
            # 성능 설정
            'enable_multiprocessing': True,
            'chunk_size': 1000,       # 데이터 청크 크기
            'cache_enabled': True,    # 캐시 활성화
            
            # 결과 저장 설정
            'save_results': True,
            'save_format': 'json',    # json, csv, excel
            'save_charts': True,
            'save_detailed_log': True,
            
            # 복리 설정
            'compound_mode': 'daily',      # daily, weekly, monthly, continuous
            'compound_ratio': 0.2,         # 복리 비율 (20%)
            'compound_threshold': 0.05,    # 복리 적용 임계값
        }
        
    def _init_ml_settings(self):
        """ML 모델 설정 초기화"""
        self.ml = {
            # 모델 설정
            'model_type': 'xgboost',      # xgboost, lightgbm, catboost
            'prediction_horizon': 24,      # 예측 호라이즌 (시간)
            'feature_window': 168,         # 피처 윈도우 (7일)
            'update_frequency': 'daily',   # 모델 업데이트 빈도
            
            # 피처 엔지니어링
            'technical_indicators': [
                'sma', 'ema', 'rsi', 'macd', 'bollinger_bands',
                'atr', 'adx', 'stochastic', 'williams_r', 'cci'
            ],
            'price_features': [
                'returns', 'log_returns', 'volatility', 'volume_profile'
            ],
            'market_features': [
                'market_cap', 'trading_volume', 'market_dominance'
            ],
            
            # 트레이닝 설정
            'train_test_split': 0.8,       # 훈련/테스트 분할 비율
            'validation_split': 0.2,       # 검증 분할 비율
            'cross_validation_folds': 5,   # 교차 검증 폴드 수
            
            # 하이퍼파라미터
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            
            # 모델 평가
            'evaluation_metrics': [
                'accuracy', 'precision', 'recall', 'f1_score',
                'roc_auc', 'sharpe_ratio', 'information_ratio'
            ],
            
            # 모델 저장
            'model_save_path': 'ml/models/',
            'model_version_control': True,
            'auto_model_backup': True,
        }
        
    def _init_risk_management_settings(self):
        """위험 관리 설정 초기화"""
        self.risk_management = {
            # 포지션 크기 조정
            'position_sizing_method': 'kelly',  # fixed, kelly, volatility
            'max_correlation': 0.7,             # 최대 상관관계
            'max_sector_exposure': 0.3,         # 최대 섹터 노출
            
            # 동적 손절/익절
            'dynamic_stop_loss': True,
            'trailing_stop_loss': True,
            'stop_loss_atr_multiple': 2.0,      # ATR 배수
            'take_profit_atr_multiple': 3.0,    # ATR 배수
            
            # 드로다운 관리
            'max_portfolio_drawdown': 0.15,     # 최대 포트폴리오 드로다운
            'drawdown_reduction_factor': 0.5,   # 드로다운 시 포지션 감소
            'recovery_threshold': 0.05,         # 회복 임계값
            
            # 변동성 관리
            'volatility_target': 0.15,          # 목표 변동성
            'volatility_lookback': 30,          # 변동성 계산 기간
            'volatility_floor': 0.05,           # 최소 변동성
            'volatility_ceiling': 0.5,          # 최대 변동성
            
            # 리스크 지표
            'var_confidence': 0.95,             # VaR 신뢰도
            'cvar_confidence': 0.95,            # CVaR 신뢰도
            'risk_free_rate': 0.02,             # 무위험 수익률
        }
        
    def _init_notification_settings(self):
        """알림 설정 초기화"""
        self.notifications = {
            # 텔레그램 설정
            'telegram': {
                'enabled': True,
                'bot_token': '',
                'chat_id': '',
                'alerts': [
                    'trade_executed', 'stop_loss_hit', 'take_profit_hit',
                    'system_error', 'daily_report', 'weekly_report'
                ]
            },
            
            # 이메일 설정
            'email': {
                'enabled': False,
                'smtp_server': '',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'recipients': []
            },
            
            # 슬랙 설정
            'slack': {
                'enabled': False,
                'webhook_url': '',
                'channel': '#trading-alerts'
            },
            
            # 디스코드 설정
            'discord': {
                'enabled': False,
                'webhook_url': ''
            }
        }
        
    def _init_data_settings(self):
        """데이터 설정 초기화"""
        self.data = {
            # 데이터 소스
            'primary_source': 'binance',        # binance, upbit, bithumb
            'backup_sources': ['upbit', 'bithumb'],
            'data_provider': 'ccxt',            # ccxt, yfinance, alpha_vantage
            
            # 심볼 설정
            'symbols': [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT',
                'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT'
            ],
            'base_currency': 'USDT',
            'quote_currencies': ['USDT', 'BTC', 'ETH'],
            
            # 데이터 수집 설정
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'primary_timeframe': '1h',
            'data_limit': 1000,                 # 데이터 수집 제한
            'update_interval': 60,              # 업데이트 간격 (초)
            
            # 데이터 저장
            'storage_format': 'parquet',        # csv, parquet, hdf5
            'compression': 'snappy',            # snappy, gzip, brotli
            'data_retention_days': 365,         # 데이터 보관 기간
            
            # 데이터 품질
            'enable_data_validation': True,
            'outlier_detection': True,
            'missing_data_strategy': 'interpolate',  # drop, interpolate, forward_fill
            
            # 실시간 데이터
            'enable_realtime': True,
            'websocket_enabled': True,
            'max_reconnection_attempts': 5,
        }
        
    def _init_dashboard_settings(self):
        """대시보드 설정 초기화"""
        self.dashboard = {
            # 서버 설정
            'host': '0.0.0.0',
            'port': 8050,
            'debug': True,
            'auto_reload': True,
            
            # 인증 설정
            'auth_enabled': False,
            'username': 'admin',
            'password': 'password',
            'session_timeout': 3600,            # 세션 타임아웃 (초)
            
            # 차트 설정
            'chart_theme': 'plotly_dark',       # plotly, plotly_white, plotly_dark
            'default_chart_height': 600,
            'update_interval': 5000,            # 업데이트 간격 (밀리초)
            
            # 성능 설정
            'max_data_points': 10000,           # 최대 데이터 포인트
            'enable_caching': True,
            'cache_timeout': 300,               # 캐시 타임아웃 (초)
            
            # 백테스트 설정
            'max_concurrent_backtests': 3,      # 최대 동시 백테스트 수
            'backtest_timeout': 300,            # 백테스트 타임아웃 (초)
        }
        
    def get_config(self, section: str = None) -> Dict[str, Any]:
        """설정 반환"""
        if section is None:
            return {
                'basic': self.basic,
                'trading': self.trading,
                'backtest': self.backtest,
                'ml': self.ml,
                'risk_management': self.risk_management,
                'notifications': self.notifications,
                'data': self.data,
                'dashboard': self.dashboard
            }
        else:
            return getattr(self, section, {})
            
    def update_config(self, section: str, settings: Dict[str, Any]):
        """설정 업데이트"""
        if hasattr(self, section):
            current_config = getattr(self, section)
            current_config.update(settings)
            
    def save_config(self, file_path: str = None):
        """설정 파일 저장"""
        if file_path is None:
            file_path = self.project_root / 'config' / 'config.json'
            
        config_dict = self.get_config()
        
        # datetime 객체를 문자열로 변환
        config_dict['backtest']['start_date'] = config_dict['backtest']['start_date'].isoformat()
        config_dict['backtest']['end_date'] = config_dict['backtest']['end_date'].isoformat()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
    def load_config(self, file_path: str = None):
        """설정 파일 로드"""
        if file_path is None:
            file_path = self.project_root / 'config' / 'config.json'
            
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                
            # 각 섹션 업데이트
            for section, settings in config_dict.items():
                if section == 'backtest':
                    # datetime 문자열을 datetime 객체로 변환
                    settings['start_date'] = datetime.fromisoformat(settings['start_date'])
                    settings['end_date'] = datetime.fromisoformat(settings['end_date'])
                    
                self.update_config(section, settings)
                
    def get_phase_config(self, phase: str) -> Dict[str, Any]:
        """Phase별 설정 반환"""
        base_config = self.trading.copy()
        
        if phase == 'aggressive':
            base_config.update({
                'max_position_size': 0.15,
                'default_leverage': 3.0,
                'default_stop_loss': 0.03,
                'default_take_profit': 0.08,
                'max_daily_loss': 0.08,
            })
        elif phase == 'defensive':
            base_config.update({
                'max_position_size': 0.08,
                'default_leverage': 1.5,
                'default_stop_loss': 0.015,
                'default_take_profit': 0.04,
                'max_daily_loss': 0.03,
            })
        elif phase == 'conservative':
            base_config.update({
                'max_position_size': 0.05,
                'default_leverage': 1.0,
                'default_stop_loss': 0.01,
                'default_take_profit': 0.025,
                'max_daily_loss': 0.02,
            })
            
        return base_config
        
    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """심볼별 설정 반환"""
        base_config = self.trading.copy()
        
        # 심볼별 특별 설정
        if symbol.startswith('BTC'):
            base_config.update({
                'default_leverage': 2.0,
                'default_stop_loss': 0.015,
            })
        elif symbol.startswith('ETH'):
            base_config.update({
                'default_leverage': 2.5,
                'default_stop_loss': 0.02,
            })
        elif symbol in ['ADA/USDT', 'DOT/USDT']:
            base_config.update({
                'default_leverage': 1.8,
                'default_stop_loss': 0.025,
            })
            
        return base_config
        
    def validate_config(self) -> List[str]:
        """설정 검증"""
        errors = []
        
        # 기본 설정 검증
        if self.trading['initial_capital'] <= 0:
            errors.append("초기 자본이 0보다 작거나 같습니다.")
            
        if self.trading['max_position_size'] > 1.0:
            errors.append("최대 포지션 크기가 100%를 초과합니다.")
            
        if self.trading['max_leverage'] > 10.0:
            errors.append("최대 레버리지가 10배를 초과합니다.")
            
        # 백테스트 설정 검증
        if self.backtest['start_date'] >= self.backtest['end_date']:
            errors.append("백테스트 시작일이 종료일보다 늦습니다.")
            
        # 데이터 설정 검증
        if not self.data['symbols']:
            errors.append("거래 심볼이 설정되지 않았습니다.")
            
        return errors

# 전역 설정 인스턴스
config = UnifiedConfig()

# 기존 호환성을 위한 별칭
backtest_config = config  # 기존 backtest_config 대신 사용