#!/usr/bin/env python3
"""
로컬용 개선된 데이터 다운로더
안정적인 데이터 수집을 위한 개선 버전
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

class LocalDataDownloaderFixed:
    """로컬용 개선된 데이터 다운로더"""
    
    def __init__(self, hours_back=24):
        self.data_path = Path("data/market_data")
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.log_path = Path("logs")
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_path / 'local_download_fixed.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 바이낸스 거래소 초기화
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
        # 로컬용 설정 - 더 안정적인 기간
        self.timeframes = ['1m', '5m', '1h']  # 다양한 시간프레임
        self.symbols = ['BTC/USDT', 'ETH/USDT']  # 주요 심볼 2개
        
        # 더 안정적인 기간 설정 (현재 시간보다 1시간 전까지)
        self.end_date = datetime.now() - timedelta(hours=1)  # 1시간 전
        self.start_date = self.end_date - timedelta(hours=hours_back)  # 24시간 전부터
        
        self.logger.info("🏠 로컬용 개선된 데이터 다운로더 초기화 완료")
        self.logger.info(f"📅 기간: {self.start_date.strftime('%Y-%m-%d %H:%M')} ~ {self.end_date.strftime('%Y-%m-%d %H:%M')}")
        self.logger.info(f"⏰ 총 {hours_back}시간의 데이터 수집")
        self.logger.info(f"🎯 심볼: {self.symbols}")
        self.logger.info(f"📊 시간프레임: {self.timeframes}")

    def test_api_connection(self):
        """API 연결 테스트"""
        try:
            self.logger.info("🔗 바이낸스 API 연결 테스트 중...")
            
            # 간단한 API 호출 테스트
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            self.logger.info(f"✅ API 연결 성공! BTC 현재가: ${ticker['last']:,.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ API 연결 실패: {str(e)}")
            return False

    def download_ohlcv_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """OHLCV 데이터 다운로드 (개선된 버전)"""
        
        try:
            self.logger.info(f"🔄 {symbol} {timeframe} 다운로드 시작...")
            
            # 시간 설정
            since = int(self.start_date.timestamp() * 1000)
            end_time = int(self.end_date.timestamp() * 1000)
            
            all_data = []
            current_time = since
            
            # 시간프레임별 한 번에 가져올 수량 설정
            limit_map = {
                '1m': 1000,   # 1000분 = 16.7시간
                '5m': 1000,   # 5000분 = 83.3시간
                '1h': 1000    # 1000시간 = 41.7일
            }
            
            limit = limit_map.get(timeframe, 1000)
            
            # 배치로 데이터 가져오기
            batch_count = 0
            while current_time < end_time:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=current_time,
                        limit=limit
                    )
                    
                    if not ohlcv:
                        self.logger.warning(f"   ⚠️ {symbol} {timeframe} 더 이상 데이터 없음")
                        break
                    
                    all_data.extend(ohlcv)
                    batch_count += 1
                    
                    # 다음 시작 시간 계산
                    last_timestamp = ohlcv[-1][0]
                    current_time = last_timestamp + self._get_timeframe_ms(timeframe)
                    
                    # 진행 상황 로그
                    last_time = datetime.fromtimestamp(last_timestamp / 1000)
                    self.logger.info(f"   📊 {symbol} {timeframe} 배치 {batch_count}: {len(ohlcv)}개, 마지막 시간: {last_time.strftime('%m-%d %H:%M')}")
                    
                    # API 제한 방지
                    time.sleep(self.exchange.rateLimit / 1000)
                    
                    # 요청한 기간을 벗어나면 중단
                    if last_timestamp >= end_time:
                        break
                        
                except Exception as e:
                    self.logger.error(f"   ❌ {symbol} {timeframe} 배치 {batch_count} 오류: {str(e)}")
                    break
            
            if not all_data:
                self.logger.warning(f"⚠️ {symbol} {timeframe} 데이터 없음")
                return pd.DataFrame()
            
            # 데이터프레임 생성
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 중복 제거 및 정렬
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            # 기간 필터링
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            
            # 파일 저장
            file_path = self.data_path / f"{symbol.replace('/', '_')}_{timeframe}.csv"
            df.to_csv(file_path)
            
            # 상세 정보 로그
            if not df.empty:
                first_time = df.index[0].strftime('%Y-%m-%d %H:%M')
                last_time = df.index[-1].strftime('%Y-%m-%d %H:%M')
                self.logger.info(f"✅ {symbol} {timeframe} 완료: {len(df):,}개 레코드")
                self.logger.info(f"   📅 데이터 범위: {first_time} ~ {last_time}")
                self.logger.info(f"   💾 저장 경로: {file_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} {timeframe} 다운로드 실패: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _get_timeframe_ms(self, timeframe: str) -> int:
        """시간프레임을 밀리초로 변환"""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '1h': 60 * 60 * 1000,
        }
        return timeframe_map.get(timeframe, 60 * 1000)

    def download_all_data(self):
        """모든 데이터 다운로드"""
        
        # API 연결 테스트
        if not self.test_api_connection():
            self.logger.error("❌ API 연결 실패로 다운로드를 중단합니다.")
            return []
        
        self.logger.info("🚀 로컬용 개선된 데이터 다운로드 시작!")
        start_time = datetime.now()
        
        results = []
        total_tasks = len(self.symbols) * len(self.timeframes)
        
        with tqdm(total=total_tasks, desc="로컬 데이터 다운로드", ncols=100) as pbar:
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    df = self.download_ohlcv_data(symbol, timeframe)
                    
                    results.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'records': len(df),
                        'success': not df.empty,
                        'file_size': f"{len(df) * 6 * 8 / 1024:.1f} KB" if not df.empty else "0 KB"  # 대략적 크기
                    })
                    
                    if not df.empty:
                        pbar.set_postfix_str(f"✅ {symbol} {timeframe} ({len(df)} records)")
                    else:
                        pbar.set_postfix_str(f"❌ {symbol} {timeframe} (0 records)")
                    
                    pbar.update(1)
                    
                    # API 제한 방지
                    time.sleep(2)
        
        # 결과 요약
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        self.logger.info("🎉 로컬 데이터 다운로드 완료!")
        self.logger.info(f"⏱️ 소요시간: {elapsed}")
        self.logger.info(f"✅ 성공: {len(successful)}개")
        self.logger.info(f"❌ 실패: {len(failed)}개")
        
        # 성공한 다운로드 상세 정보
        total_records = sum(r['records'] for r in successful)
        self.logger.info(f"📊 총 레코드: {total_records:,}개")
        
        print("\n" + "="*80)
        print("📊 다운로드 결과 상세:")
        print("="*80)
        
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"{status} {result['symbol']} {result['timeframe']}: {result['records']:,}개 레코드 ({result['file_size']})")
        
        if failed:
            print("\n❌ 실패한 다운로드:")
            for result in failed:
                print(f"   - {result['symbol']} {result['timeframe']}")
        
        return results

def main():
    try:
        print("🏠 로컬용 개선된 데이터 다운로더")
        print("=" * 60)
        print("🔧 더 안정적이고 많은 데이터를 다운로드합니다")
        print("=" * 60)
        
        # 사용자 선택
        print("\n다운로드할 데이터 기간을 선택하세요:")
        print("1. 빠른 테스트 (1시간)")
        print("2. 표준 테스트 (24시간) - 권장")
        print("3. 확장 테스트 (7일)")
        
        choice = input("\n선택 (1-3, 기본값 2): ").strip()
        
        hours_map = {'1': 1, '2': 24, '3': 168}  # 7일 = 168시간
        hours = hours_map.get(choice, 24)
        
        print(f"\n⏰ {hours}시간 데이터 다운로드를 시작합니다...")
        
        downloader = LocalDataDownloaderFixed(hours_back=hours)
        results = downloader.download_all_data()
        
        print("\n" + "=" * 60)
        print("🎉 개선된 데이터 다운로드 완료!")
        print("이제 안정적인 백테스트를 실행할 수 있습니다.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 다운로드 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 