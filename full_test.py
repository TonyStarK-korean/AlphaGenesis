#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대시보드와 백테스트 통합 테스트 스크립트
"""

import subprocess
import time
import requests
import threading
from datetime import datetime

def start_dashboard():
    """대시보드를 별도 프로세스로 실행"""
    try:
        print("🚀 대시보드 실행 중...")
        subprocess.Popen(['python', 'dashboard/simple_dashboard.py'], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
        
        # 대시보드 시작 대기
        time.sleep(3)
        
        # 대시보드 연결 테스트
        for attempt in range(5):
            try:
                response = requests.get('http://localhost:5000')
                if response.status_code == 200:
                    print("✅ 대시보드 연결 성공!")
                    return True
            except:
                print(f"⏳ 대시보드 연결 시도 {attempt+1}/5...")
                time.sleep(2)
        
        print("❌ 대시보드 연결 실패")
        return False
        
    except Exception as e:
        print(f"❌ 대시보드 실행 오류: {e}")
        return False

def send_test_log(message):
    """테스트 로그를 대시보드로 전송"""
    try:
        response = requests.post(
            'http://localhost:5000/api/realtime_log',
            json={'log': message},
            timeout=2
        )
        return response.status_code == 200
    except:
        return False

def run_simple_test():
    """간단한 연동 테스트"""
    print("📊 연동 테스트 시작...")
    
    # 테스트 데이터
    test_messages = [
        "[2024-01-01 12:00:00] === 매매 현황 === | 총자산: 10000000 | 실현손익: +50000 | 미실현손익: +100000 | 수익률: +1.50% | 보유포지션: 2개",
        "[2024-01-01 12:00:01] 추세추종 전략 | 상승 시장국면 | ML예측: +5.25% | 현재가: 50000원",
        "[2024-01-01 12:00:02] | 진입 | 상승 | 추세추종 | 매수 | BTC/USDT | 50000.00 | 51000.00 | +2.00% | +50000 | 10150000 | 10.0% | 2.50배 | 추세추종 조건충족 | ML예측: +5.25%",
        "[2024-01-01 12:00:03] === 매매 현황 === | 총자산: 10150000 | 실현손익: +100000 | 미실현손익: +50000 | 수익률: +2.00% | 보유포지션: 3개",
        "[2024-01-01 12:00:04] 역추세 전략 | 급락 시장국면 | ML예측: -8.75% | 현재가: 49000원"
    ]
    
    for i, message in enumerate(test_messages):
        print(f"[{i+1}/5] 테스트 로그 전송...")
        success = send_test_log(message)
        if success:
            print(f"✅ 성공: {message[:50]}...")
        else:
            print(f"❌ 실패: {message[:50]}...")
        time.sleep(2)
    
    print("✅ 연동 테스트 완료!")
    print("🌐 브라우저에서 http://localhost:5000 을 열어서 실시간 데이터를 확인하세요!")

if __name__ == '__main__':
    print("🎯 AlphaGenesis 대시보드 연동 테스트")
    print("=" * 50)
    
    # 1. 대시보드 실행
    if start_dashboard():
        # 2. 연동 테스트 실행
        run_simple_test()
        
        print("\n📝 테스트 완료!")
        print("대시보드를 계속 사용하시려면 Ctrl+C로 종료하세요.")
        
        # 대시보드 계속 실행
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 테스트 종료!")
    else:
        print("❌ 대시보드 실행 실패 - 테스트를 종료합니다.") 