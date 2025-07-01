#!/usr/bin/env python3
"""
로컬용 20분 데이터 다운로더
빠른 테스트를 위한 최소 데이터 수집
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
from tqdm import tqdm
import json

class LocalDataDownloader:
    """로컬용 20분 데이터 다운로더"""
    
    def __init__(self):
        self.data_path = Path("data/market_data")
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.log_path = Path("logs")
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_path / 'local_download.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 바이낸스 거래소 초기화
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
        # 로컬용 설정 (20분치 데이터)
        self.timeframes = ['1m', '5m']  # 빠른 테스트용 최소 시간프레임
        self.symbols = ['BTC/USDT']     # 테스트용 주요 심볼만
        
        # 20분 기간 설정
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(minutes=20)
        
        self.logger.info("🏠 로컬용 20분 데이터 다운로더 초기화 완료")
        self.logger.info(f"📅 기간: {self.start_date.strftime('%Y-%m-%d %H:%M')} ~ {self.end_date.strftime('%Y-%m-%d %H:%M')}")
        self.logger.info(f"⏰ 총 20분간의 데이터 수집")

    def download_ohlcv_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """OHLCV 데이터 다운로드"""
        
        try:
            self.logger.info(f"🔄 {symbol} {timeframe} 다운로드 시작...")
            
            # 20분치 데이터 가져오기
            since = int(self.start_date.timestamp() * 1000)
            limit = 50  # 20분이므로 적은 수량
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            
            if not ohlcv:
                self.logger.warning(f"⚠️ {symbol} {timeframe} 데이터 없음")
                return pd.DataFrame()
            
            # 데이터프레임 생성
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 중복 제거 및 정렬
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            # 20분 기간으로 필터링
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            
            # 파일 저장
            file_path = self.data_path / f"{symbol.replace('/', '_')}_{timeframe}.csv"
            df.to_csv(file_path)
            
            self.logger.info(f"✅ {symbol} {timeframe} 완료: {len(df)}개 레코드")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} {timeframe} 다운로드 실패: {str(e)}")
            return pd.DataFrame()

    def download_all_data(self):
        """모든 20분 데이터 다운로드"""
        
        self.logger.info("🚀 로컬용 20분 데이터 다운로드 시작!")
        start_time = datetime.now()
        
        results = []
        total_tasks = len(self.symbols) * len(self.timeframes)
        
        with tqdm(total=total_tasks, desc="로컬 데이터 다운로드", ncols=80) as pbar:
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    df = self.download_ohlcv_data(symbol, timeframe)
                    
                    results.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'records': len(df),
                        'success': not df.empty
                    })
                    
                    pbar.set_postfix_str(f"{symbol} {timeframe}")
                    pbar.update(1)
                    
                    # API 제한 방지
                    time.sleep(1)
        
        # 결과 요약
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        self.logger.info("🎉 로컬 20분 데이터 다운로드 완료!")
        self.logger.info(f"⏱️ 소요시간: {elapsed}")
        self.logger.info(f"✅ 성공: {len(successful)}개")
        self.logger.info(f"❌ 실패: {len(failed)}개")
        
        # 성공한 다운로드 상세 정보
        total_records = sum(r['records'] for r in successful)
        self.logger.info(f"📊 총 레코드: {total_records:,}개")
        
        for result in successful:
            self.logger.info(f"   📈 {result['symbol']} {result['timeframe']}: {result['records']:,}개")
        
        return results

def main():
    try:
        print("🏠 로컬용 20분 데이터 다운로더")
        print("=" * 50)
        
        downloader = LocalDataDownloader()
        results = downloader.download_all_data()
        
        print("\n" + "=" * 50)
        print("🎉 20분 데이터 다운로드 완료!")
        print("이제 빠른 백테스트를 실행할 수 있습니다.")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 다운로드 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 