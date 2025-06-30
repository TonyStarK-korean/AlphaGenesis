#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GVS 서버용 백테스트 실행 스크립트
실시간 대시보드 연동 포함
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import time
import json
from datetime import datetime
import threading
import subprocess

def send_log_to_dashboard(log_msg):
    """대시보드로 로그 전송 (로컬과 원격 모두)"""
    urls = [
        'http://localhost:5000/api/realtime_log',    # 로컬 테스트용
        'http://34.47.77.230:5000/api/realtime_log'  # 원격 대시보드
    ]
    
    for url in urls:
        try:
            response = requests.post(
                url, 
                json={'log': log_msg}, 
                timeout=3
            )
            if response.status_code == 200:
                print(f"✅ 대시보드 전송 성공 ({url})")
                break
        except Exception as e:
            print(f"❌ 대시보드 전송 실패 ({url}): {e}")
            continue

def check_dashboard_connection():
    """대시보드 연결 상태 확인"""
    print("🔍 대시보드 연결 상태 확인 중...")
    
    urls = [
        ('로컬', 'http://localhost:5000'),
        ('원격', 'http://34.47.77.230:5000')
    ]
    
    for name, url in urls:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name} 대시보드 연결 성공: {url}")
            else:
                print(f"❌ {name} 대시보드 연결 실패: {response.status_code}")
        except Exception as e:
            print(f"❌ {name} 대시보드 연결 오류: {e}")

def run_ml_backtest():
    """ML 백테스트 실행"""
    print("🚀 ML 백테스트 실행 시작...")
    
    # 백테스트 실행 전 대시보드 연결 확인
    check_dashboard_connection()
    
    # 시작 알림 전송
    start_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 GVS 서버에서 ML 백테스트 시작!"
    send_log_to_dashboard(start_msg)
    
    try:
        # ML 백테스트 실행
        print("📊 run_ml_backtest.py 실행 중...")
        
        # Python 스크립트 실행 (실시간 출력 캡처)
        process = subprocess.Popen(
            ['python', 'run_ml_backtest.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 실시간 출력 처리
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output = output.strip()
                print(output)  # 서버 콘솔에 출력
                
                # 대시보드로 전송 (중요한 로그만)
                if any(keyword in output for keyword in [
                    '매매 현황', 'ML예측', '진입', '청산', '전략', '시장국면',
                    '총자산', '수익률', '실현손익', '미실현손익', '보유포지션'
                ]):
                    send_log_to_dashboard(output)
        
        # 프로세스 완료 대기
        return_code = process.poll()
        
        if return_code == 0:
            end_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 백테스트 완료!"
            print(end_msg)
            send_log_to_dashboard(end_msg)
        else:
            error_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 백테스트 오류 (코드: {return_code})"
            print(error_msg)
            send_log_to_dashboard(error_msg)
        
    except Exception as e:
        error_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 백테스트 실행 오류: {e}"
        print(error_msg)
        send_log_to_dashboard(error_msg)

def run_parallel_backtest():
    """병렬 백테스트 실행"""
    print("🚀 병렬 백테스트 실행 시작...")
    
    # 대시보드 연결 확인
    check_dashboard_connection()
    
    # 시작 알림 전송
    start_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 GVS 서버에서 병렬 백테스트 시작!"
    send_log_to_dashboard(start_msg)
    
    try:
        # 병렬 백테스트 실행
        print("📊 run_parallel_backtest.py 실행 중...")
        
        process = subprocess.Popen(
            ['python', 'run_parallel_backtest.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 실시간 출력 처리
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output = output.strip()
                print(output)  # 서버 콘솔에 출력
                
                # 대시보드로 전송
                if any(keyword in output for keyword in [
                    'BTC', 'ETH', 'XRP', 'BNB', 'ADA', 'DOT',
                    '완료', '시작', '처리', '백테스트'
                ]):
                    send_log_to_dashboard(output)
        
        return_code = process.poll()
        
        if return_code == 0:
            end_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 병렬 백테스트 완료!"
            print(end_msg)
            send_log_to_dashboard(end_msg)
        else:
            error_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 병렬 백테스트 오류 (코드: {return_code})"
            print(error_msg)
            send_log_to_dashboard(error_msg)
        
    except Exception as e:
        error_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 병렬 백테스트 실행 오류: {e}"
        print(error_msg)
        send_log_to_dashboard(error_msg)

def main():
    """메인 실행 함수"""
    print("🎯 GVS 서버 백테스트 실행기")
    print("=" * 50)
    print("1. ML 백테스트")
    print("2. 병렬 백테스트")
    print("3. 연결 테스트만")
    print("=" * 50)
    
    try:
        choice = input("선택하세요 (1-3): ").strip()
        
        if choice == '1':
            run_ml_backtest()
        elif choice == '2':
            run_parallel_backtest()
        elif choice == '3':
            check_dashboard_connection()
            # 테스트 로그 전송
            test_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🧪 GVS 서버 연결 테스트"
            send_log_to_dashboard(test_msg)
        else:
            print("❌ 잘못된 선택입니다.")
    
    except KeyboardInterrupt:
        print("\n👋 사용자가 중단했습니다.")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")

if __name__ == '__main__':
    main() 