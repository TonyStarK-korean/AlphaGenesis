#!/usr/bin/env python3
"""
3개월치 데이터 다운로더 (로컬용)
백테스트를 위한 3개월 데이터 수집
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
from concurrent.futures import ThreadPoolExecutor, as_completed

class ThreeMonthDownloader:
    """3개월치 데이터 다운로더"""
    
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
                logging.FileHandler(self.log_path / f'three_month_download_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 바이낸스 거래소 초기화
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
        # 3개월치 설정
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
        
        # 3개월 기간 설정 (2시간 전까지 - 안정적인 데이터)
        self.end_date = datetime.now() - timedelta(hours=2)
        self.start_date = self.end_date - timedelta(days=90)  # 3개월 = 90일
        
        self.logger.info("📅 3개월치 데이터 다운로더 초기화 완료")
        self.logger.info(f"📊 기간: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"⏰ 총 90일(3개월)의 데이터 수집")
        self.logger.info(f"🎯 심볼: {self.symbols}")
        self.logger.info(f"📈 시간프레임: {self.timeframes}")

    def calculate_expected_records(self, timeframe: str) -> int:
        """예상 레코드 수 계산"""
        days = 90  # 3개월
        
        records_per_day = {
            '1m': 1440,    # 1일 = 1440분
            '5m': 288,     # 1일 = 288개 5분봉
            '15m': 96,     # 1일 = 96개 15분봉
            '1h': 24,      # 1일 = 24시간
            '4h': 6,       # 1일 = 6개 4시간봉
            '1d': 1        # 1일 = 1개 일봉
        }
        
        return records_per_day.get(timeframe, 24) * days

    def test_api_connection(self):
        """API 연결 테스트"""
        try:
            self.logger.info("🔗 바이낸스 API 연결 테스트 중...")
            
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            self.logger.info(f"✅ API 연결 성공! BTC 현재가: ${ticker['last']:,.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ API 연결 실패: {str(e)}")
            return False

    def download_single_timeframe(self, symbol: str, timeframe: str) -> dict:
        """단일 시간프레임 데이터 다운로드"""
        
        try:
            self.logger.info(f"🔄 {symbol} {timeframe} 3개월치 다운로드 시작...")
            
            # 예상 레코드 수
            expected_records = self.calculate_expected_records(timeframe)
            self.logger.info(f"   📊 예상 레코드: {expected_records:,}개")
            
            # 파일 경로
            file_path = self.data_path / f"{symbol.replace('/', '_')}_{timeframe}.csv"
            
            # 시간 설정
            since = int(self.start_date.timestamp() * 1000)
            end_time = int(self.end_date.timestamp() * 1000)
            
            all_data = []
            current_time = since
            
            # 시간프레임별 배치 크기 최적화
            limit_map = {
                '1m': 1000,   # 1000분 = 16.7시간
                '5m': 1000,   # 5000분 = 83.3시간
                '15m': 1000,  # 15000분 = 250시간
                '1h': 1000,   # 1000시간 = 41.7일
                '4h': 1000,   # 4000시간 = 166.7일
                '1d': 1000    # 1000일 = 2.7년
            }
            
            limit = limit_map.get(timeframe, 1000)
            batch_count = 0
            total_records = 0
            
            # 진행 상황 표시용
            progress_bar = tqdm(
                total=expected_records, 
                desc=f"{symbol} {timeframe}", 
                unit="records",
                leave=False
            )
            
            while current_time < end_time:
                try:
                    # 데이터 가져오기
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=current_time,
                        limit=limit
                    )
                    
                    if not ohlcv:
                        self.logger.warning(f"   ⚠️ {symbol} {timeframe} 더 이상 데이터 없음")
                        break
                    
                    # 데이터 추가
                    all_data.extend(ohlcv)
                    batch_count += 1
                    total_records += len(ohlcv)
                    
                    # 진행 상황 업데이트
                    progress_bar.update(len(ohlcv))
                    
                    # 다음 시작 시간 계산
                    last_timestamp = ohlcv[-1][0]
                    current_time = last_timestamp + self._get_timeframe_ms(timeframe)
                    
                    # 중간 진행 상황 로그 (10배치마다)
                    if batch_count % 10 == 0:
                        last_time = datetime.fromtimestamp(last_timestamp / 1000)
                        progress_percent = (total_records / expected_records * 100) if expected_records > 0 else 0
                        self.logger.info(f"   📊 {symbol} {timeframe}: {total_records:,}/{expected_records:,} ({progress_percent:.1f}%), 현재 {last_time.strftime('%Y-%m-%d')}")
                    
                    # API 제한 방지
                    time.sleep(self.exchange.rateLimit / 1000)
                    
                    # 요청한 기간을 벗어나면 중단
                    if last_timestamp >= end_time:
                        break
                        
                except Exception as e:
                    self.logger.error(f"   ❌ {symbol} {timeframe} 배치 {batch_count} 오류: {str(e)}")
                    time.sleep(5)  # 오류 시 대기
                    continue
            
            progress_bar.close()
            
            if not all_data:
                return {'symbol': symbol, 'timeframe': timeframe, 'success': False, 'records': 0, 'error': 'No data'}
            
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
            df.to_csv(file_path)
            
            # 결과 정보
            if not df.empty:
                first_time = df.index[0].strftime('%Y-%m-%d %H:%M')
                last_time = df.index[-1].strftime('%Y-%m-%d %H:%M')
                completion_rate = (len(df) / expected_records * 100) if expected_records > 0 else 0
                
                self.logger.info(f"✅ {symbol} {timeframe} 완료: {len(df):,}/{expected_records:,}개 ({completion_rate:.1f}%)")
                self.logger.info(f"   📅 실제 범위: {first_time} ~ {last_time}")
                self.logger.info(f"   💾 저장: {file_path}")
            
            return {
                'symbol': symbol, 
                'timeframe': timeframe, 
                'success': True, 
                'records': len(df),
                'expected': expected_records,
                'completion_rate': completion_rate,
                'file_path': str(file_path),
                'start_date': df.index[0].strftime('%Y-%m-%d') if not df.empty else None,
                'end_date': df.index[-1].strftime('%Y-%m-%d') if not df.empty else None
            }
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} {timeframe} 다운로드 실패: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'symbol': symbol, 'timeframe': timeframe, 'success': False, 'error': str(e)}

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
        return timeframe_map.get(timeframe, 60 * 1000)

    def download_all_data(self, parallel=True):
        """모든 3개월 데이터 다운로드"""
        
        # API 연결 테스트
        if not self.test_api_connection():
            self.logger.error("❌ API 연결 실패로 다운로드를 중단합니다.")
            return []
        
        self.logger.info("🚀 3개월치 전체 데이터 다운로드 시작!")
        start_time = datetime.now()
        
        # 다운로드 작업 리스트
        tasks = []
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                tasks.append((symbol, timeframe))
        
        self.logger.info(f"📋 총 {len(tasks)}개 작업 (심볼 {len(self.symbols)}개 × 시간프레임 {len(self.timeframes)}개)")
        
        results = []
        
        if parallel:
            # 병렬 다운로드 (빠름, 하지만 API 제한 주의)
            self.logger.info("⚡ 병렬 다운로드 모드")
            with ThreadPoolExecutor(max_workers=3) as executor:  # 3개 동시 실행
                future_to_task = {
                    executor.submit(self.download_single_timeframe, symbol, timeframe): (symbol, timeframe)
                    for symbol, timeframe in tasks
                }
                
                with tqdm(total=len(tasks), desc="3개월 데이터 다운로드", ncols=120) as pbar:
                    for future in as_completed(future_to_task):
                        symbol, timeframe = future_to_task[future]
                        try:
                            result = future.result()
                            results.append(result)
                            
                            if result['success']:
                                pbar.set_postfix_str(f"✅ {symbol} {timeframe} ({result['records']:,} records)")
                            else:
                                pbar.set_postfix_str(f"❌ {symbol} {timeframe}")
                                
                        except Exception as e:
                            self.logger.error(f"작업 예외 {symbol} {timeframe}: {str(e)}")
                            results.append({
                                'symbol': symbol, 
                                'timeframe': timeframe, 
                                'success': False, 
                                'error': str(e)
                            })
                        
                        pbar.update(1)
        else:
            # 순차 다운로드 (안정적)
            self.logger.info("🔄 순차 다운로드 모드")
            with tqdm(total=len(tasks), desc="3개월 데이터 다운로드", ncols=120) as pbar:
                for symbol, timeframe in tasks:
                    result = self.download_single_timeframe(symbol, timeframe)
                    results.append(result)
                    
                    if result['success']:
                        pbar.set_postfix_str(f"✅ {symbol} {timeframe} ({result['records']:,} records)")
                    else:
                        pbar.set_postfix_str(f"❌ {symbol} {timeframe}")
                    
                    pbar.update(1)
                    time.sleep(1)  # 안전 간격
        
        # 결과 요약
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        self.logger.info("🎉 3개월 데이터 다운로드 완료!")
        self.logger.info(f"⏱️ 소요시간: {elapsed}")
        self.logger.info(f"✅ 성공: {len(successful)}개")
        self.logger.info(f"❌ 실패: {len(failed)}개")
        
        # 성공한 다운로드 상세 정보
        total_records = sum(r['records'] for r in successful)
        total_expected = sum(r.get('expected', 0) for r in successful)
        overall_completion = (total_records / total_expected * 100) if total_expected > 0 else 0
        
        self.logger.info(f"📊 총 레코드: {total_records:,}/{total_expected:,}개 ({overall_completion:.1f}%)")
        
        # 결과 상세 출력
        print("\n" + "="*100)
        print("📊 3개월 데이터 다운로드 결과 상세:")
        print("="*100)
        
        for result in results:
            if result['success']:
                status = "✅"
                info = f"{result['records']:,}/{result.get('expected', 0):,}개 ({result.get('completion_rate', 0):.1f}%)"
                date_range = f"({result.get('start_date', '')} ~ {result.get('end_date', '')})"
            else:
                status = "❌"
                info = f"실패: {result.get('error', 'Unknown error')}"
                date_range = ""
            
            print(f"{status} {result['symbol']:>10} {result['timeframe']:>4}: {info} {date_range}")
        
        # 실패 요약
        if failed:
            print(f"\n❌ 실패한 {len(failed)}개 다운로드:")
            for result in failed:
                print(f"   - {result['symbol']} {result['timeframe']}: {result.get('error', 'Unknown')}")
        
        # 성공률 요약
        success_rate = len(successful) / len(results) * 100
        print(f"\n📈 전체 성공률: {success_rate:.1f}% ({len(successful)}/{len(results)})")
        print("="*100)
        
        return results

def main():
    try:
        print("📅 3개월치 데이터 다운로더")
        print("=" * 70)
        print("🎯 백테스트를 위한 3개월 완전 데이터 수집")
        print("=" * 70)
        
        # 다운로드 모드 선택
        print("\n다운로드 모드를 선택하세요:")
        print("1. 병렬 다운로드 (빠름, 권장)")
        print("2. 순차 다운로드 (안정적)")
        
        mode_choice = input("\n선택 (1-2, 기본값 1): ").strip()
        parallel_mode = mode_choice != '2'
        
        mode_name = "병렬" if parallel_mode else "순차"
        print(f"\n⚡ {mode_name} 모드로 3개월 데이터 다운로드를 시작합니다...")
        
        downloader = ThreeMonthDownloader()
        results = downloader.download_all_data(parallel=parallel_mode)
        
        successful_count = len([r for r in results if r['success']])
        total_records = sum(r['records'] for r in results if r['success'])
        
        print("\n" + "=" * 70)
        print("🎉 3개월 데이터 다운로드 완료!")
        print(f"📊 총 {total_records:,}개 레코드 수집 완료")
        print("이제 완전한 백테스트를 실행할 수 있습니다.")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ 다운로드 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 