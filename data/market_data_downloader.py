"""
시장 데이터 다운로더
설정된 날짜 범위에 따라 거래소 데이터를 다운로드
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
import os
from pathlib import Path
from tqdm import tqdm

# 설정 파일 임포트
import sys
sys.path.append('../../')
from config.backtest_config import backtest_config

class MarketDataDownloader:
    """시장 데이터 다운로더"""
    
    def __init__(self):
        self.exchanges = {}
        self.data_path = Path("data/market_data")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 거래소 초기화
        self._initialize_exchanges()
        
    def _initialize_exchanges(self):
        """거래소 초기화"""
        exchange_configs = {
            'binance': ccxt.binance,
            'upbit': ccxt.upbit,
            'bithumb': ccxt.bithumb,
            'coinone': ccxt.coinone
        }
        
        for exchange_name, exchange_class in exchange_configs.items():
            try:
                exchange = exchange_class({
                    'apiKey': backtest_config.exchanges[exchange_name]['api_key'],
                    'secret': backtest_config.exchanges[exchange_name]['secret_key'],
                    'enableRateLimit': True,
                    'timeout': 30000
                })
                self.exchanges[exchange_name] = exchange
                self.logger.info(f"{exchange_name} 거래소 초기화 완료")
            except Exception as e:
                self.logger.warning(f"{exchange_name} 거래소 초기화 실패: {str(e)}")
                
    def download_ohlcv_data(self, 
                           symbol: str, 
                           timeframe: str = '1h',
                           start_date: datetime = None,
                           end_date: datetime = None,
                           exchange_name: str = 'binance') -> pd.DataFrame:
        """OHLCV 데이터 다운로드"""
        
        if exchange_name not in self.exchanges:
            raise ValueError(f"거래소 {exchange_name}이 초기화되지 않았습니다.")
            
        exchange = self.exchanges[exchange_name]
        
        # 기본 날짜 설정
        if start_date is None:
            start_date = backtest_config.start_date
        if end_date is None:
            end_date = backtest_config.end_date
            
        # 시간프레임을 밀리초로 변환
        timeframe_ms = self._get_timeframe_ms(timeframe)
        
        # 데이터 수집
        all_data = []
        current_date = start_date
        
        while current_date < end_date:
            try:
                # 거래소에서 데이터 가져오기
                ohlcv = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=int(current_date.timestamp() * 1000),
                    limit=1000
                )
                
                if not ohlcv:
                    break
                    
                # 데이터프레임으로 변환
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                all_data.append(df)
                
                # 다음 시작 시간 계산
                if len(ohlcv) > 0:
                    last_timestamp = ohlcv[-1][0]
                    current_date = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=timeframe_ms)
                else:
                    break
                    
                # API 제한 방지
                time.sleep(exchange.rateLimit / 1000)
                
            except Exception as e:
                self.logger.error(f"데이터 다운로드 오류 ({symbol}, {current_date}): {str(e)}")
                break
                
        if not all_data:
            return pd.DataFrame()
            
        # 모든 데이터 합치기
        combined_df = pd.concat(all_data)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df.sort_index(inplace=True)
        
        # 설정된 기간으로 필터링
        combined_df = combined_df[(combined_df.index >= start_date) & (combined_df.index <= end_date)]
        
        return combined_df
        
    def _get_timeframe_ms(self, timeframe: str) -> int:
        """시간프레임을 밀리초로 변환"""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return timeframe_map.get(timeframe, 60 * 60 * 1000)
        
    def download_all_data(self) -> Dict[str, pd.DataFrame]:
        """모든 설정된 심볼의 데이터 다운로드"""
        
        all_data = {}
        symbols = backtest_config.data_download['symbols']
        timeframe = backtest_config.data_download['timeframe']
        
        self.logger.info(f"데이터 다운로드 시작: {len(symbols)}개 심볼, {timeframe} 시간프레임")
        
        for symbol in tqdm(symbols, desc='데이터 다운로드 진행중', ncols=80):
            try:
                self.logger.info(f"{symbol} 데이터 다운로드 중...")
                
                # 파일 경로 설정
                file_path = self.data_path / f"{symbol.replace('/', '_')}_{timeframe}.csv"
                
                # 기존 파일이 있으면 로드
                if file_path.exists():
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    self.logger.info(f"{symbol} 기존 데이터 로드: {len(df)}개 레코드")
                else:
                    # 새로 다운로드
                    df = self.download_ohlcv_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=backtest_config.start_date,
                        end_date=backtest_config.end_date
                    )
                    
                    # 파일 저장
                    if not df.empty:
                        df.to_csv(file_path)
                        self.logger.info(f"{symbol} 데이터 저장: {len(df)}개 레코드")
                        
                all_data[symbol] = df
                
            except Exception as e:
                self.logger.error(f"{symbol} 데이터 다운로드 실패: {str(e)}")
                continue
                
        self.logger.info(f"데이터 다운로드 완료: {len(all_data)}개 심볼")
        return all_data
        
    def get_market_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """특정 심볼의 시장 데이터 반환"""
        
        timeframe = backtest_config.data_download['timeframe']
        file_path = self.data_path / f"{symbol.replace('/', '_')}_{timeframe}.csv"
        
        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # 날짜 필터링
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            return df
        else:
            self.logger.warning(f"{symbol} 데이터 파일이 없습니다. 다운로드를 실행하세요.")
            return pd.DataFrame()
            
    def update_data(self, symbol: str, days_back: int = 7):
        """최신 데이터 업데이트"""
        
        timeframe = backtest_config.data_download['timeframe']
        file_path = self.data_path / f"{symbol.replace('/', '_')}_{timeframe}.csv"
        
        if file_path.exists():
            # 기존 데이터 로드
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            last_date = df.index[-1]
            
            # 최신 데이터 다운로드
            start_date = last_date + timedelta(milliseconds=self._get_timeframe_ms(timeframe))
            end_date = datetime.now()
            
            new_df = self.download_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if not new_df.empty:
                # 기존 데이터와 합치기
                combined_df = pd.concat([df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                combined_df.sort_index(inplace=True)
                
                # 파일 저장
                combined_df.to_csv(file_path)
                self.logger.info(f"{symbol} 데이터 업데이트 완료: {len(new_df)}개 새 레코드")
                
        else:
            # 파일이 없으면 전체 다운로드
            self.download_ohlcv_data(symbol=symbol, timeframe=timeframe)
            
    def get_data_summary(self) -> Dict:
        """데이터 요약 정보 반환"""
        
        summary = {}
        symbols = backtest_config.data_download['symbols']
        timeframe = backtest_config.data_download['timeframe']
        
        for symbol in symbols:
            file_path = self.data_path / f"{symbol.replace('/', '_')}_{timeframe}.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                summary[symbol] = {
                    'records': len(df),
                    'start_date': df.index[0].strftime('%Y-%m-%d'),
                    'end_date': df.index[-1].strftime('%Y-%m-%d'),
                    'file_size': f"{file_path.stat().st_size / 1024:.1f} KB"
                }
            else:
                summary[symbol] = {
                    'records': 0,
                    'start_date': 'N/A',
                    'end_date': 'N/A',
                    'file_size': 'N/A'
                }
                
        return summary 