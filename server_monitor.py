#!/usr/bin/env python3
"""
서버 다운로드 진행 상황 모니터링
"""

import json
import time
from datetime import datetime
from pathlib import Path
import os

def monitor_server_download():
    """서버 다운로드 진행 상황 모니터링"""
    
    log_path = Path("logs")
    status_file = log_path / "server_3month_status.json"
    
    print("🔍 서버 3개월 데이터 다운로드 모니터링 시작...")
    print("Ctrl+C를 누르면 종료됩니다.\n")
    
    try:
        while True:
            # 화면 클리어
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 100)
            print("🌐 서버 3개월 데이터 다운로드 실시간 모니터링")
            print("🖥️ 서버 IP: 34.47.77.230")
            print("=" * 100)
            print(f"📍 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            if status_file.exists():
                try:
                    with open(status_file, 'r', encoding='utf-8') as f:
                        status = json.load(f)
                    
                    stats = status.get('stats', {})
                    
                    # 기본 정보
                    start_time = datetime.fromisoformat(stats.get('start_time', datetime.now().isoformat()))
                    elapsed = datetime.now() - start_time
                    current_phase = stats.get('current_phase', 'Unknown')
                    
                    print(f"⏰ 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"⏱️ 경과 시간: {format_duration(elapsed.total_seconds())}")
                    print(f"🔄 현재 단계: {current_phase}")
                    print()
                    
                    # OHLCV 진행 상황
                    ohlcv_completed = stats.get('ohlcv_completed', 0)
                    total_ohlcv = stats.get('total_ohlcv_tasks', 30)
                    ohlcv_progress = (ohlcv_completed / total_ohlcv * 100) if total_ohlcv > 0 else 0
                    
                    print("📊 OHLCV 분봉 데이터:")
                    print(f"   진행률: {ohlcv_progress:.1f}% ({ohlcv_completed}/{total_ohlcv})")
                    
                    if current_phase == 'ohlcv_download':
                        print(f"   상태: 🔄 다운로드 중...")
                    elif ohlcv_completed >= total_ohlcv:
                        print(f"   상태: ✅ 완료")
                    else:
                        print(f"   상태: ⏳ 대기 중...")
                    
                    print()
                    
                    # 틱데이터 진행 상황
                    tick_progress = stats.get('tick_progress', {})
                    print("⚡ 틱데이터:")
                    
                    if tick_progress:
                        for symbol, progress in tick_progress.items():
                            total_ticks = progress.get('total_ticks', 0)
                            file_count = progress.get('file_count', 0)
                            last_update = progress.get('last_update', '')
                            
                            print(f"   🔸 {symbol}:")
                            print(f"      틱 수: {total_ticks:,}개")
                            print(f"      파일: {file_count}개")
                            if last_update:
                                update_time = datetime.fromisoformat(last_update)
                                print(f"      업데이트: {update_time.strftime('%H:%M:%S')}")
                    else:
                        if current_phase == 'tick_download':
                            print(f"   상태: 🔄 시작 중...")
                        else:
                            print(f"   상태: ⏳ 대기 중...")
                    
                    print()
                    
                    # 오류 정보
                    errors = stats.get('errors', [])
                    if errors:
                        print(f"⚠️ 최근 오류: {len(errors)}개")
                        for error in errors[-3:]:
                            error_time = datetime.fromisoformat(error['timestamp'])
                            symbol = error.get('symbol', 'Unknown')
                            err_msg = error.get('error', 'Unknown error')[:50]
                            print(f"   🔸 {error_time.strftime('%H:%M:%S')} - {symbol}: {err_msg}...")
                    else:
                        print("✅ 오류 없음")
                    
                    print()
                    
                    # 데이터 파일 현황
                    data_path = Path("data/market_data")
                    tick_path = Path("data/tick_data")
                    
                    if data_path.exists():
                        ohlcv_files = list(data_path.glob("*.csv"))
                        total_size = sum(f.stat().st_size for f in ohlcv_files)
                        print(f"📁 OHLCV 파일: {len(ohlcv_files)}개 ({format_size(total_size)})")
                    
                    if tick_path.exists():
                        tick_files = list(tick_path.glob("*.pkl.gz"))
                        total_tick_size = sum(f.stat().st_size for f in tick_files)
                        print(f"📁 틱데이터 파일: {len(tick_files)}개 ({format_size(total_tick_size)})")
                    
                except Exception as e:
                    print(f"❌ 상태 파일 읽기 오류: {str(e)}")
            else:
                print("⏳ 서버 다운로드가 아직 시작되지 않았습니다...")
            
            print()
            print("=" * 100)
            
            time.sleep(10)  # 10초마다 업데이트
            
    except KeyboardInterrupt:
        print("\n🛑 모니터링 종료")

def format_duration(seconds):
    """초를 읽기 쉬운 형태로 변환"""
    if seconds < 60:
        return f"{seconds:.0f}초"
    elif seconds < 3600:
        return f"{seconds/60:.1f}분"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}시간"
    else:
        return f"{seconds/86400:.1f}일"

def format_size(bytes_size):
    """바이트를 읽기 쉬운 형태로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

if __name__ == "__main__":
    monitor_server_download() 