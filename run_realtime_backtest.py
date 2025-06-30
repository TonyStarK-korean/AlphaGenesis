#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaGenesis 실시간 백테스트 & 대시보드 통합 실행 스크립트
"""

import os
import sys
import threading
import time
import subprocess
import webbrowser

def start_dashboard_background():
    """백그라운드에서 대시보드 실행"""
    try:
        from dashboard.simple_dashboard import app
        print("📊 대시보드 서버 시작중...")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"❌ 대시보드 오류: {e}")

def start_backtest():
    """백테스트 실행"""
    try:
        from run_ml_backtest import main
        print("🎯 백테스트 시작...")
        main()
    except Exception as e:
        print(f"❌ 백테스트 오류: {e}")

def open_browser_after_delay():
    """5초 후 브라우저 열기"""
    time.sleep(5)
    try:
        webbrowser.open('http://localhost:5000')
        print("🌐 대시보드가 브라우저에서 열렸습니다!")
    except:
        print("🌐 브라우저를 수동으로 열어주세요: http://localhost:5000")

def main():
    print("🚀 AlphaGenesis 실시간 백테스트 시스템 시작!")
    print("=" * 70)
    print("📊 로컬 대시보드: http://localhost:5000")
    print("🌐 외부 대시보드: http://34.47.77.230:5000")
    print("🎯 백테스트: 실시간 연동")
    print("🔄 업데이트: 1초마다")
    print("=" * 70)
    
    # 1. 대시보드 백그라운드 실행
    dashboard_thread = threading.Thread(target=start_dashboard_background, daemon=True)
    dashboard_thread.start()
    
    # 2. 브라우저 자동 열기 (5초 후)
    browser_thread = threading.Thread(target=open_browser_after_delay, daemon=True)
    browser_thread.start()
    
    # 3. 잠시 대기 후 백테스트 실행
    print("⏳ 대시보드 준비 중... (3초 대기)")
    time.sleep(3)
    
    print("🎯 백테스트를 시작합니다!")
    print("📈 실시간 결과를 대시보드에서 확인하세요!")
    print("=" * 70)
    
    # 4. 백테스트 실행 (메인 스레드에서)
    start_backtest()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 시스템이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("개별 실행을 시도하세요:")
        print("1. python start_dashboard.py")
        print("2. python run_ml_backtest.py") 