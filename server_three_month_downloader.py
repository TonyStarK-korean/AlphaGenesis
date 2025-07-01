#!/usr/bin/env python3
"""
서버용 3개월 데이터 다운로더
터미널 종료 후에도 백그라운드에서 계속 실행
분봉 + 틱데이터 동시 수집
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import json
import signal
import threading
import gzip
import pickle
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class ServerThreeMonthDownloader:
    """서버용 3개월 데이터 다운로더 (분봉 + 틱데이터)"""
    
    def __init__(self):
        self.data_path = Path("data/market_data")
        self.tick_path = Path("data/tick_data")
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.tick_path.mkdir(parents=True, exist_ok=True)
        self.log_path = Path("logs")
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # 상태 파일들
        self.status_file = self.log_path / "server_3month_status.json"
        self.progress_file = self.log_path / "server_3month_progress.json"
        
        # 로깅 설정
        self.setup_logging()
        
        # 바이낸스 거래소 초기화
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
        # 서버용 설정
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']  # 분봉 데이터
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
        self.tick_symbols = ['BTC/USDT', 'ETH/USDT']  # 틱데이터는 주요 심볼만
        
        # 3개월 기간 설정
        self.end_date = datetime.now() - timedelta(hours=2)
        self.start_date = self.end_date - timedelta(days=90)  # 3개월
        
        # 상태 관리
        self.is_running = True
        self.stats = {
            'start_time': datetime.now(),
            'ohlcv_completed': 0,
            'tick_progress': {},
            'total_ohlcv_tasks': len(self.symbols) * len(self.timeframes),
            'total_tick_tasks': len(self.tick_symbols),
            'current_phase': 'initializing',  # initializing, ohlcv_download, tick_download, completed
            'last_update': datetime.now(),
            'errors': []
        }
        
        # 시그널 핸들러 등록 (안전한 종료)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("🌐 서버용 3개월 데이터 다운로더 초기화 완료")
        self.logger.info(f"📅 기간: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"🎯 분봉 심볼: {self.symbols}")
        self.logger.info(f"⚡ 틱데이터 심볼: {self.tick_symbols}")
        self.logger.info(f"📊 서버 IP: 34.47.77.230")

    def setup_logging(self):
        """로깅 설정"""
        log_file = self.log_path / f"server_3month_{datetime.now().strftime('%Y%m%d')}.log"
        
        # 로그 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def signal_handler(self, signum, frame):
        """시그널 핸들러 (안전한 종료)"""
        self.logger.info(f"🛑 종료 신호 수신 (Signal {signum})")
        self.is_running = False
        self.save_status()
        self.logger.info("💾 상태 저장 완료 - 안전하게 종료 중...")

    def save_status(self):
        """현재 상태 저장 (재시작 가능하도록)"""
        try:
            status_data = {
                'timestamp': datetime.now().isoformat(),
                'is_running': self.is_running,
                'server_ip': '34.47.77.230',
                'stats': {
                    'start_time': self.stats['start_time'].isoformat(),
                    'ohlcv_completed': self.stats['ohlcv_completed'],
                    'tick_progress': self.stats['tick_progress'],
                    'total_ohlcv_tasks': self.stats['total_ohlcv_tasks'],
                    'total_tick_tasks': self.stats['total_tick_tasks'],
                    'current_phase': self.stats['current_phase'],
                    'last_update': self.stats['last_update'].isoformat(),
                    'errors': self.stats['errors'][-50:]  # 최근 50개 오류만
                }
            }
            
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"상태 저장 오류: {str(e)}")

    def save_progress(self, progress_data):
        """진행 상황 저장"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"진행 상황 저장 오류: {str(e)}")

    def test_api_connection(self):
        """API 연결 테스트"""
        try:
            self.logger.info("🔗 서버에서 바이낸스 API 연결 테스트 중...")
            
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            self.logger.info(f"✅ 서버 API 연결 성공! BTC 현재가: ${ticker['last']:,.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 서버 API 연결 실패: {str(e)}")
            return False

    def download_single_timeframe(self, symbol: str, timeframe: str) -> dict:
        """단일 시간프레임 데이터 다운로드"""
        
        try:
            self.logger.info(f"🔄 [서버] {symbol} {timeframe} 3개월치 다운로드 시작...")
            
            # 파일 경로
            file_path = self.data_path / f"{symbol.replace('/', '_')}_{timeframe}.csv"
            
            # 시간 설정
            since = int(self.start_date.timestamp() * 1000)
            end_time = int(self.end_date.timestamp() * 1000)
            
            all_data = []
            current_time = since
            
            # 시간프레임별 최적화된 리미트
            limit_map = {
                '1m': 1000,   '5m': 1000,   '15m': 1000,
                '1h': 1000,   '4h': 1000,   '1d': 1000
            }
            
            limit = limit_map.get(timeframe, 1000)
            batch_count = 0
            total_records = 0
            
            while current_time < end_time and self.is_running:
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
                    
                    # 다음 시작 시간
                    last_timestamp = ohlcv[-1][0]
                    current_time = last_timestamp + self._get_timeframe_ms(timeframe)
                    
                    # 진행 상황 로그 (서버에서는 더 자주)
                    if batch_count % 5 == 0:
                        last_time = datetime.fromtimestamp(last_timestamp / 1000)
                        self.logger.info(f"   📊 [서버] {symbol} {timeframe}: {total_records:,}개, 현재 {last_time.strftime('%Y-%m-%d')}")
                    
                    # API 제한 방지
                    time.sleep(self.exchange.rateLimit / 1000)
                    
                    # 요청한 기간을 벗어나면 중단
                    if last_timestamp >= end_time:
                        break
                        
                except Exception as e:
                    self.logger.error(f"   ❌ {symbol} {timeframe} 배치 {batch_count} 오류: {str(e)}")
                    self.stats['errors'].append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'error': str(e)
                    })
                    time.sleep(5)
                    continue
            
            if not all_data:
                return {'symbol': symbol, 'timeframe': timeframe, 'success': False, 'records': 0}
            
            # 데이터프레임 생성 및 저장
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
            
            # 성공 로그
            if not df.empty:
                first_time = df.index[0].strftime('%Y-%m-%d')
                last_time = df.index[-1].strftime('%Y-%m-%d')
                self.logger.info(f"✅ [서버] {symbol} {timeframe} 완료: {len(df):,}개 레코드 ({first_time} ~ {last_time})")
            
            return {
                'symbol': symbol, 
                'timeframe': timeframe, 
                'success': True, 
                'records': len(df),
                'file_path': str(file_path)
            }
            
        except Exception as e:
            self.logger.error(f"❌ [서버] {symbol} {timeframe} 다운로드 실패: {str(e)}")
            return {'symbol': symbol, 'timeframe': timeframe, 'success': False, 'error': str(e)}

    def _get_timeframe_ms(self, timeframe: str) -> int:
        """시간프레임을 밀리초로 변환"""
        timeframe_map = {
            '1m': 60 * 1000,        '5m': 5 * 60 * 1000,     '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,   '4h': 4 * 60 * 60 * 1000, '1d': 24 * 60 * 60 * 1000
        }
        return timeframe_map.get(timeframe, 60 * 1000)

    def download_all_ohlcv(self):
        """모든 OHLCV 데이터 다운로드"""
        
        self.logger.info("🚀 [서버] 3개월 OHLCV 데이터 다운로드 시작!")
        self.stats['current_phase'] = 'ohlcv_download'
        self.save_status()
        
        start_time = datetime.now()
        
        # 다운로드 작업 리스트
        tasks = []
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                tasks.append((symbol, timeframe))
        
        self.logger.info(f"📋 총 {len(tasks)}개 OHLCV 작업")
        
        # 병렬 다운로드 (서버에서는 조금 더 보수적으로)
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_task = {
                executor.submit(self.download_single_timeframe, symbol, timeframe): (symbol, timeframe)
                for symbol, timeframe in tasks
            }
            
            for future in as_completed(future_to_task):
                if not self.is_running:
                    break
                    
                symbol, timeframe = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        self.stats['ohlcv_completed'] += 1
                        progress = (self.stats['ohlcv_completed'] / self.stats['total_ohlcv_tasks'] * 100)
                        self.logger.info(f"   ✅ [서버] {symbol} {timeframe} ({self.stats['ohlcv_completed']}/{self.stats['total_ohlcv_tasks']}, {progress:.1f}%)")
                    else:
                        self.logger.error(f"   ❌ [서버] {symbol} {timeframe} 실패")
                    
                    # 진행 상황 저장
                    self.stats['last_update'] = datetime.now()
                    self.save_status()
                    
                except Exception as e:
                    self.logger.error(f"작업 예외 {symbol} {timeframe}: {str(e)}")
                    results.append({
                        'symbol': symbol, 
                        'timeframe': timeframe, 
                        'success': False, 
                        'error': str(e)
                    })
        
        # OHLCV 완료 요약
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        elapsed = datetime.now() - start_time
        
        self.logger.info(f"🎉 [서버] OHLCV 다운로드 완료! 소요시간: {elapsed}")
        self.logger.info(f"✅ 성공: {len(successful)}개, ❌ 실패: {len(failed)}개")
        
        return results

    def download_tick_data_continuous(self, symbol: str):
        """틱데이터 연속 다운로드 (3개월치)"""
        
        self.logger.info(f"⚡ [서버] {symbol} 틱데이터 3개월치 다운로드 시작")
        
        # 심볼별 상태 초기화
        if symbol not in self.stats['tick_progress']:
            self.stats['tick_progress'][symbol] = {
                'last_timestamp': None,
                'total_ticks': 0,
                'file_count': 0,
                'start_time': datetime.now().isoformat(),
                'last_update': datetime.now().isoformat()
            }
        
        symbol_stats = self.stats['tick_progress'][symbol]
        current_batch = []
        
        # 3개월 전부터 시작
        since = int(self.start_date.timestamp() * 1000)
        end_time = int(self.end_date.timestamp() * 1000)
        
        if symbol_stats['last_timestamp']:
            since = symbol_stats['last_timestamp']
        
        while self.is_running and since < end_time:
            try:
                # 거래 데이터 가져오기
                trades = self.exchange.fetch_trades(symbol=symbol, since=since, limit=1000)
                
                if not trades:
                    time.sleep(10)
                    continue
                
                # 배치에 추가
                current_batch.extend(trades)
                symbol_stats['total_ticks'] += len(trades)
                
                # 마지막 타임스탬프 업데이트
                last_trade = trades[-1]
                since = last_trade['timestamp'] + 1
                symbol_stats['last_timestamp'] = since
                
                # 저장 조건 (10,000개마다)
                if len(current_batch) >= 10000:
                    self.save_tick_batch(symbol, current_batch, symbol_stats['file_count'])
                    current_batch = []
                    symbol_stats['file_count'] += 1
                    symbol_stats['last_update'] = datetime.now().isoformat()
                    
                    # 진행 상황 로그
                    current_date = datetime.fromtimestamp(last_trade['timestamp'] / 1000)
                    self.logger.info(f"⚡ [서버] {symbol} 틱: {symbol_stats['total_ticks']:,}개, 현재 {current_date.strftime('%Y-%m-%d')}")
                
                # API 제한 방지
                time.sleep(self.exchange.rateLimit / 1000)
                
                # 상태 주기적 저장 (5분마다)
                if symbol_stats['total_ticks'] % 50000 == 0:
                    self.save_status()
                
            except Exception as e:
                self.logger.error(f"[서버] {symbol} 틱데이터 오류: {str(e)}")
                self.stats['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'type': 'tick_data',
                    'error': str(e)
                })
                time.sleep(30)
                continue
        
        # 마지막 배치 저장
        if current_batch:
            self.save_tick_batch(symbol, current_batch, symbol_stats['file_count'])
            symbol_stats['file_count'] += 1
        
        symbol_stats['last_update'] = datetime.now().isoformat()
        self.logger.info(f"🏁 [서버] {symbol} 틱데이터 완료: {symbol_stats['total_ticks']:,}개")

    def save_tick_batch(self, symbol: str, trades: list, file_index: int):
        """틱데이터 배치 저장"""
        try:
            symbol_clean = symbol.replace('/', '_')
            date_str = datetime.now().strftime('%Y%m%d')
            file_name = f"{symbol_clean}_ticks_{date_str}_{file_index:06d}.pkl.gz"
            file_path = self.tick_path / file_name
            
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(trades, f)
            
            self.logger.info(f"💾 [서버] {symbol} 틱데이터 저장: {len(trades):,}개 -> {file_name}")
            
        except Exception as e:
            self.logger.error(f"틱데이터 저장 오류 {symbol}: {str(e)}")

    def run_full_download(self):
        """전체 다운로드 실행 (OHLCV + 틱데이터)"""
        
        # API 연결 테스트
        if not self.test_api_connection():
            self.logger.error("❌ 서버 API 연결 실패로 다운로드를 중단합니다.")
            return
        
        self.logger.info("🌐 서버용 3개월 전체 데이터 다운로드 시작!")
        
        try:
            # Phase 1: OHLCV 데이터 다운로드
            self.logger.info("📊 Phase 1: OHLCV 3개월 데이터 다운로드")
            ohlcv_results = self.download_all_ohlcv()
            
            if not self.is_running:
                self.logger.info("🛑 사용자 요청으로 중단됨")
                return
            
            # Phase 2: 틱데이터 다운로드 (백그라운드)
            self.logger.info("⚡ Phase 2: 틱데이터 3개월 다운로드 시작")
            self.stats['current_phase'] = 'tick_download'
            self.save_status()
            
            tick_threads = []
            for symbol in self.tick_symbols:
                thread = threading.Thread(
                    target=self.download_tick_data_continuous,
                    args=(symbol,),
                    name=f"ServerTickDownloader-{symbol.replace('/', '_')}"
                )
                thread.daemon = True
                tick_threads.append(thread)
                thread.start()
                time.sleep(2)  # 스레드 시작 간격
            
            # 틱데이터 다운로드 모니터링
            while self.is_running and any(t.is_alive() for t in tick_threads):
                time.sleep(300)  # 5분마다 체크
                
                # 진행 상황 로그
                total_ticks = sum(stats['total_ticks'] for stats in self.stats['tick_progress'].values())
                self.logger.info(f"⚡ [서버] 틱데이터 진행 중... 총 {total_ticks:,}개")
                
                # 상태 저장
                self.save_status()
            
            # 모든 스레드 대기
            for thread in tick_threads:
                thread.join(timeout=60)
            
            # 완료
            self.stats['current_phase'] = 'completed'
            self.save_status()
            
            self.logger.info("🎉 [서버] 3개월 전체 데이터 다운로드 완료!")
            
            # 최종 요약
            successful_ohlcv = len([r for r in ohlcv_results if r['success']])
            total_ticks = sum(stats['total_ticks'] for stats in self.stats['tick_progress'].values())
            
            self.logger.info("📊 최종 요약:")
            self.logger.info(f"   OHLCV 성공: {successful_ohlcv}/{len(ohlcv_results)}")
            self.logger.info(f"   틱데이터: {total_ticks:,}개")
            
        except Exception as e:
            self.logger.error(f"[서버] 전체 다운로드 오류: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.save_status()

def main():
    try:
        print("🌐 서버용 3개월 데이터 다운로더")
        print("=" * 70)
        print("📊 분봉 + 틱데이터 동시 수집 (터미널 종료 후에도 계속 실행)")
        print("🖥️ 서버 IP: 34.47.77.230")
        print("=" * 70)
        
        downloader = ServerThreeMonthDownloader()
        downloader.run_full_download()
        
    except Exception as e:
        print(f"❌ 서버 다운로드 오류: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 