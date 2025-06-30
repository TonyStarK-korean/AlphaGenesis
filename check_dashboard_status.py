#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대시보드 API 상태 확인 스크립트
"""

import requests
import json
from datetime import datetime

def check_dashboard_status():
    """대시보드 상태 확인"""
    print("🔍 대시보드 상태 확인 중...")
    print("=" * 50)
    
    try:
        # API 호출
        response = requests.get('http://localhost:5000/api/realtime_data', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ 대시보드 연결 성공!")
            print("\n📊 현재 실시간 데이터:")
            print("-" * 30)
            print(f"총자산: {data.get('current_capital', 0):,.0f}원")
            print(f"수익률: {data.get('total_return', 0):+.2f}%")
            print(f"실현손익: {data.get('realized_pnl', 0):+,.0f}원")
            print(f"미실현손익: {data.get('unrealized_pnl', 0):+,.0f}원")
            print(f"보유포지션: {data.get('open_positions', 0)}개")
            print(f"거래횟수: {data.get('trades_count', 0)}회")
            print(f"현재전략: {data.get('strategy', 'N/A')}")
            print(f"시장국면: {data.get('regime', 'N/A')}")
            print(f"ML예측: {data.get('ml_prediction', 0):+.2f}%")
            
            last_update = data.get('last_update')
            if last_update:
                print(f"마지막 업데이트: {last_update}")
            
            # 로그 개수 확인
            logs_count = len(data.get('logs', []))
            print(f"저장된 로그: {logs_count}개")
            
            # 차트 데이터 확인
            capital_history = data.get('capital_history', [])
            if capital_history:
                print(f"차트 데이터: {len(capital_history)}개 포인트")
                print(f"최근 자산: {capital_history[-3:] if len(capital_history) > 3 else capital_history}")
            
            print("-" * 30)
            print("🌐 브라우저에서 http://localhost:5000 접속하여 실시간 차트를 확인하세요!")
            
        else:
            print(f"❌ API 호출 실패: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == '__main__':
    check_dashboard_status() 