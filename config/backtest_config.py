"""
백테스트 설정 파일
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

class BacktestConfig:
    """백테스트 설정 클래스"""
    
    def __init__(self):
        # 기본 백테스트 기간 설정
        self.start_date = datetime(2023, 1, 1)  # 시작일
        self.end_date = datetime(2024, 12, 31)  # 종료일
        
        # 거래 설정
        self.initial_capital = 10_000_000  # 1000만원
        self.target_capital = 100_000_000  # 1억원
        self.max_position_size = 0.1  # 최대 포지션 크기 (10%)
        self.default_stop_loss = 0.02  # 기본 손절 (2%)
        self.default_take_profit = 0.05  # 기본 익절 (5%)
        
        # 복리 설정
        self.compound_mode = 'daily'  # 복리 모드: daily, weekly, monthly, continuous
        self.compound_ratio = 0.2  # 복리 비율 (20%)
        
        # Phase 설정
        self.phase1_aggressive = {
            'leverage': 3.0,  # 레버리지
            'position_size': 0.15,  # 포지션 크기
            'stop_loss': 0.03,  # 손절
            'take_profit': 0.08,  # 익절
            'target_coins': ['BTC', 'ETH', 'BNB', 'ADA', 'DOT'],  # 대상 코인
            'strategy': 'momentum_breakout'  # 전략
        }
        
        self.phase2_defensive = {
            'leverage': 1.5,  # 레버리지
            'position_size': 0.08,  # 포지션 크기
            'stop_loss': 0.015,  # 손절
            'take_profit': 0.04,  # 익절
            'target_coins': ['BTC', 'ETH', 'USDT', 'USDC'],  # 대상 코인
            'strategy': 'mean_reversion'  # 전략
        }
        
        # 시장 국면 분석 설정
        self.market_analysis = {
            'volatility_threshold': 0.05,  # 변동성 임계값
            'trend_period': 20,  # 트렌드 분석 기간
            'rsi_period': 14,  # RSI 기간
            'rsi_oversold': 30,  # RSI 과매도
            'rsi_overbought': 70,  # RSI 과매수
            'volume_threshold': 1.5  # 거래량 임계값
        }
        
        # 자동 Phase 전환 조건
        self.phase_transition = {
            'aggressive_to_defensive': {
                'consecutive_losses': 3,  # 연속 손실 횟수
                'drawdown_threshold': 0.15,  # 낙폭 임계값
                'market_volatility': 0.08,  # 시장 변동성
                'rsi_condition': 'overbought'  # RSI 조건
            },
            'defensive_to_aggressive': {
                'consecutive_wins': 5,  # 연속 승리 횟수
                'profit_threshold': 0.05,  # 수익 임계값
                'market_volatility': 0.03,  # 시장 변동성
                'rsi_condition': 'oversold'  # RSI 조건
            }
        }
        
        # 다중 거래소 설정
        self.exchanges = {
            'binance': {
                'enabled': True,
                'weight': 0.4,  # 가중치
                'api_key': '',  # API 키
                'secret_key': ''  # 시크릿 키
            },
            'upbit': {
                'enabled': True,
                'weight': 0.3,
                'api_key': '',
                'secret_key': ''
            },
            'bithumb': {
                'enabled': True,
                'weight': 0.2,
                'api_key': '',
                'secret_key': ''
            },
            'coinone': {
                'enabled': True,
                'weight': 0.1,
                'api_key': '',
                'secret_key': ''
            }
        }
        
        # 데이터 다운로드 설정
        self.data_download = {
            'timeframe': '1h',  # 시간프레임: 1m, 5m, 15m, 1h, 4h, 1d
            'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT'],
            'limit': 1000,  # 데이터 개수
            'auto_download': True  # 자동 다운로드
        }
        
        # 백테스트 결과 저장 설정
        self.results = {
            'save_to_file': True,
            'file_format': 'json',  # json, csv, excel
            'include_charts': True,
            'save_path': 'results/'
        }
        
    def update_date_range(self, start_date: str, end_date: str):
        """날짜 범위 업데이트"""
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
    def update_phase_settings(self, phase: str, settings: Dict):
        """Phase 설정 업데이트"""
        if phase == 'phase1':
            self.phase1_aggressive.update(settings)
        elif phase == 'phase2':
            self.phase2_defensive.update(settings)
            
    def update_market_analysis(self, settings: Dict):
        """시장 분석 설정 업데이트"""
        self.market_analysis.update(settings)
        
    def update_exchange_settings(self, exchange: str, settings: Dict):
        """거래소 설정 업데이트"""
        if exchange in self.exchanges:
            self.exchanges[exchange].update(settings)
            
    def get_config_summary(self) -> Dict:
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
            'data_download': self.data_download
        }

# 전역 설정 인스턴스
backtest_config = BacktestConfig() 