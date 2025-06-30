#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 대시보드 연동 테스트
"""

import requests
import time
from datetime import datetime

def test_dashboard_connection():
    """대시보드 연결 테스트"""
    try:
        print("🔍 대시보드 연결 테스트 중...")
        response = requests.get('http://localhost:5000', timeout=5)
        if response.status_code == 200:
            print("✅ 대시보드 연결 성공!")
            return True
        else:
            print(f"❌ 대시보드 연결 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 대시보드 연결 오류: {e}")
        return False

def send_test_log(message):
    """테스트 로그 전송"""
    try:
        response = requests.post(
            'http://localhost:5001/api/realtime_log',
            json={'log': message},
            timeout=5
        )
        if response.status_code == 200:
            print(f"✅ 로그 전송 성공: {message[:50]}...")
            return True
        else:
            print(f"❌ 로그 전송 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 로그 전송 오류: {e}")
        return False

def run_test():
    """테스트 실행"""
    print("🎯 AlphaGenesis 대시보드 연동 테스트")
    print("=" * 50)
    
    # 1. 대시보드 연결 확인
    if not test_dashboard_connection():
        print("❌ 대시보드가 실행되지 않았습니다.")
        print("먼저 다음 명령으로 대시보드를 실행하세요:")
        print("python dashboard/simple_dashboard.py")
        return
    
    # 2. 테스트 로그 전송
    print("\n📊 테스트 로그 전송 시작...")
    
    test_logs = [
        "[2024-01-01 12:00:00] === 매매 현황 === | 총자산: 10000000 | 실현손익: +50000 | 미실현손익: +100000 | 수익률: +1.50% | 보유포지션: 2개",
        "[2024-01-01 12:00:01] 추세추종 전략 | 상승 시장국면 | ML예측: +5.25% | 현재가: 50000원",
        "[2024-01-01 12:00:02] | 진입 | 상승 | 추세추종 | 매수 | BTC/USDT | 50000.00 | 51000.00 | +2.00% | +50000 | 10150000 | 10.0% | 2.50배 | 추세추종 조건충족 | ML예측: +5.25%",
        "[2024-01-01 12:00:03] === 매매 현황 === | 총자산: 10200000 | 실현손익: +150000 | 미실현손익: +50000 | 수익률: +2.50% | 보유포지션: 3개",
        "[2024-01-01 12:00:04] 역추세 전략 | 급락 시장국면 | ML예측: -8.75% | 현재가: 49000원"
    ]
    
    for i, log in enumerate(test_logs):
        print(f"\n[{i+1}/5] 테스트 로그 전송 중...")
        if send_test_log(log):
            print("✅ 성공!")
        else:
            print("❌ 실패!")
        time.sleep(2)
    
    print("\n✅ 테스트 완료!")
    print("🌐 브라우저에서 http://localhost:5000 또는 http://34.47.77.230:5000 을 열어서 확인하세요!")

if __name__ == '__main__':
    run_test() 