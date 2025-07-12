"""
백테스트 설정 파일 - 통합 설정 시스템 사용
기존 하드코딩 제거 및 중앙 집중식 설정 관리
"""

import os
from .unified_config import UnifiedConfig, config

class BacktestConfig:
    """백테스트 설정 클래스 - 통합 설정 래퍼"""
    
    def __init__(self):
        # 통합 설정 인스턴스 사용
        self._config = config
        
        # 환경변수에서 설정 오버라이드
        self._load_from_environment()
        
    def _load_from_environment(self):
        """환경변수에서 설정 로드"""
        env_overrides = {}
        
        # 거래 설정 환경변수
        if os.getenv('INITIAL_CAPITAL'):
            env_overrides['initial_capital'] = int(os.getenv('INITIAL_CAPITAL'))
        if os.getenv('TARGET_CAPITAL'):
            env_overrides['target_capital'] = int(os.getenv('TARGET_CAPITAL'))
        if os.getenv('MAX_LEVERAGE'):
            env_overrides['max_leverage'] = float(os.getenv('MAX_LEVERAGE'))
        if os.getenv('DEFAULT_LEVERAGE'):
            env_overrides['default_leverage'] = float(os.getenv('DEFAULT_LEVERAGE'))
        if os.getenv('MAX_POSITION_SIZE'):
            env_overrides['max_position_size'] = float(os.getenv('MAX_POSITION_SIZE'))
        if os.getenv('DEFAULT_STOP_LOSS'):
            env_overrides['default_stop_loss'] = float(os.getenv('DEFAULT_STOP_LOSS'))
        if os.getenv('DEFAULT_TAKE_PROFIT'):
            env_overrides['default_take_profit'] = float(os.getenv('DEFAULT_TAKE_PROFIT'))
            
        # 백테스트 설정 환경변수
        if os.getenv('BACKTEST_START_DATE'):
            from datetime import datetime
            start_date = datetime.strptime(os.getenv('BACKTEST_START_DATE'), '%Y-%m-%d')
            self._config.update_config('backtest', {'start_date': start_date})
        if os.getenv('BACKTEST_END_DATE'):
            from datetime import datetime
            end_date = datetime.strptime(os.getenv('BACKTEST_END_DATE'), '%Y-%m-%d')
            self._config.update_config('backtest', {'end_date': end_date})
        if os.getenv('BACKTEST_TIMEFRAME'):
            self._config.update_config('backtest', {'timeframe': os.getenv('BACKTEST_TIMEFRAME')})
            
        # 거래 설정 업데이트
        if env_overrides:
            self._config.update_config('trading', env_overrides)
    
    # 기존 호환성을 위한 프로퍼티들
    @property
    def start_date(self):
        return self._config.backtest['start_date']
    
    @property
    def end_date(self):
        return self._config.backtest['end_date']
    
    @property
    def initial_capital(self):
        return self._config.trading['initial_capital']
    
    @property
    def target_capital(self):
        return self._config.trading['target_capital']
    
    @property
    def max_position_size(self):
        return self._config.trading['max_position_size']
    
    @property
    def default_stop_loss(self):
        return self._config.trading['default_stop_loss']
    
    @property
    def default_take_profit(self):
        return self._config.trading['default_take_profit']
    
    @property
    def compound_mode(self):
        return self._config.backtest['compound_mode']
    
    @property
    def compound_ratio(self):
        return self._config.backtest['compound_ratio']
    
    @property
    def phase1_aggressive(self):
        return self._config.get_phase_config('aggressive')
    
    @property
    def phase2_defensive(self):
        return self._config.get_phase_config('defensive')
    
    @property
    def market_analysis(self):
        return {
            'volatility_threshold': self._config.risk_management.get('volatility_target', 0.05),
            'trend_period': 20,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1.5
        }
    
    @property
    def phase_transition(self):
        return {
            'aggressive_to_defensive': {
                'consecutive_losses': 3,
                'drawdown_threshold': self._config.risk_management['max_portfolio_drawdown'],
                'market_volatility': 0.08,
                'rsi_condition': 'overbought'
            },
            'defensive_to_aggressive': {
                'consecutive_wins': 5,
                'profit_threshold': 0.05,
                'market_volatility': 0.03,
                'rsi_condition': 'oversold'
            }
        }
    
    @property
    def exchanges(self):
        return {
            'binance': {
                'enabled': True,
                'weight': 0.4,
                'api_key': os.getenv('BINANCE_API_KEY', ''),
                'secret_key': os.getenv('BINANCE_SECRET_KEY', '')
            },
            'upbit': {
                'enabled': True,
                'weight': 0.3,
                'api_key': os.getenv('UPBIT_API_KEY', ''),
                'secret_key': os.getenv('UPBIT_SECRET_KEY', '')
            },
            'bithumb': {
                'enabled': True,
                'weight': 0.2,
                'api_key': os.getenv('BITHUMB_API_KEY', ''),
                'secret_key': os.getenv('BITHUMB_SECRET_KEY', '')
            },
            'coinone': {
                'enabled': True,
                'weight': 0.1,
                'api_key': os.getenv('COINONE_API_KEY', ''),
                'secret_key': os.getenv('COINONE_SECRET_KEY', '')
            }
        }
    
    @property
    def data_download(self):
        return {
            'timeframe': self._config.data['primary_timeframe'],
            'symbols': self._config.data['symbols'],
            'limit': self._config.data['data_limit'],
            'auto_download': True
        }
    
    @property
    def results(self):
        return {
            'save_to_file': self._config.backtest['save_results'],
            'file_format': self._config.backtest['save_format'],
            'include_charts': self._config.backtest['save_charts'],
            'save_path': 'results/'
        }
    
    def update_date_range(self, start_date: str, end_date: str):
        """날짜 범위 업데이트"""
        from datetime import datetime
        updates = {
            'start_date': datetime.strptime(start_date, '%Y-%m-%d'),
            'end_date': datetime.strptime(end_date, '%Y-%m-%d')
        }
        self._config.update_config('backtest', updates)
        
    def update_phase_settings(self, phase: str, settings: dict):
        """Phase 설정 업데이트"""
        # 통합 설정에서 Phase별 설정은 get_phase_config로 처리
        # 여기서는 기본 거래 설정을 업데이트
        if phase in ['phase1', 'aggressive']:
            self._config.update_config('trading', settings)
        elif phase in ['phase2', 'defensive']:
            self._config.update_config('trading', settings)
            
    def update_market_analysis(self, settings: dict):
        """시장 분석 설정 업데이트"""
        self._config.update_config('risk_management', settings)
        
    def update_exchange_settings(self, exchange: str, settings: dict):
        """거래소 설정 업데이트 - 환경변수 사용 권장"""
        # 보안상 환경변수 사용 권장
        print(f"거래소 {exchange} 설정은 환경변수로 설정하세요:")
        print(f"{exchange.upper()}_API_KEY=your_api_key")
        print(f"{exchange.upper()}_SECRET_KEY=your_secret_key")
            
    def get_config_summary(self) -> dict:
        """설정 요약 반환"""
        return {
            'date_range': {
                'start': self.start_date.strftime('%Y-%m-%d'),
                'end': self.end_date.strftime('%Y-%m-%d'),
                'days': (self.end_date - self.start_date).days
            },
            'capital': {
                'initial': self.initial_capital,
                'target': self.target_capital,
                'multiplier': self.target_capital / self.initial_capital
            },
            'compound_mode': self.compound_mode,
            'phase_settings': {
                'phase1': self.phase1_aggressive,
                'phase2': self.phase2_defensive
            },
            'exchanges': {k: v['enabled'] for k, v in self.exchanges.items()},
            'data_download': self.data_download,
            'environment_loaded': {
                'initial_capital': bool(os.getenv('INITIAL_CAPITAL')),
                'api_keys': {
                    'binance': bool(os.getenv('BINANCE_API_KEY')),
                    'upbit': bool(os.getenv('UPBIT_API_KEY')),
                    'bithumb': bool(os.getenv('BITHUMB_API_KEY')),
                }
            }
        }

# 전역 설정 인스턴스
backtest_config = BacktestConfig()