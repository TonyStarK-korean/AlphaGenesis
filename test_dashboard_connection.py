#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대시보드 연동 테스트용 간단한 백테스트 시뮬레이터
"""

import requests
import time
import random
from datetime import datetime

def send_log_to_dashboard(log_msg):
    """대시보드로 로그 전송"""
    try:
        url = 'http://localhost:5001/api/realtime_log'
        response = requests.post(url, json={'log': log_msg}, timeout=2)
        print(f"✅ 로그 전송 성공: {response.status_code}")
    except Exception as e:
        print(f"❌ 로그 전송 실패: {e}")

def simulate_backtest():
    """백테스트 시뮬레이션"""
    print("🎯 테스트 백테스트 시작!")
    print("📊 대시보드 연동 테스트 중...")
    
    # 초기 설정
    initial_capital = 10000000
    current_capital = initial_capital
    realized_pnl = 0
    unrealized_pnl = 0
    open_positions = 0
    trades_count = 0
    
    # 30초간 테스트 실행
    for i in range(30):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 랜덤 데이터 생성
        price_change = random.uniform(-0.05, 0.05)  # -5% ~ +5%
        current_capital = max(current_capital * (1 + price_change), 1000000)
        
        total_pnl = current_capital - initial_capital
        total_return = (total_pnl / initial_capital) * 100
        
        # ML 예측값 (랜덤)
        ml_prediction = random.uniform(-15, 15)
        
        # 전략 랜덤 선택
        strategies = ['추세추종', '역추세', '모멘텀돌파', '숏모멘텀', '비트코인숏전략']
        strategy = random.choice(strategies)
        
        # 시장국면 랜덤 선택
        regimes = ['급등', '상승', '횡보', '하락', '급락']
        regime = random.choice(regimes)
        
        # 가끔 거래 발생
        if random.random() < 0.3:  # 30% 확률로 거래
            if random.random() < 0.6:  # 60% 확률로 진입
                trades_count += 1
                open_positions = min(open_positions + 1, 5)
                action = "진입"
                direction = "매수" if random.random() < 0.5 else "매도"
            else:  # 40% 확률로 청산
                if open_positions > 0:
                    open_positions -= 1
                    action = "청산"
                    direction = "매수" if random.random() < 0.5 else "매도"
                    # 실현손익 업데이트
                    profit = random.uniform(-50000, 100000)
                    realized_pnl += profit
                else:
                    action = None
            
            if action:
                log_msg = f"[{timestamp}] | {action:^4} | {regime:^4} | {strategy:^10} | {direction:^4} | BTC/USDT | 50000.00 | 51000.00 | +2.00% | +50000 | {current_capital:>10,.0f} | 10.0% | 2.50배 | {strategy} 조건충족 | ML예측: {ml_prediction:.2f}%"
                send_log_to_dashboard(log_msg)
        
        # 매매 현황 로그 (가끔)
        if i % 5 == 0 or open_positions > 0:
            unrealized_pnl = random.uniform(-100000, 200000) if open_positions > 0 else 0
            log_msg = f"[{timestamp}] === 매매 현황 === | 총자산: {current_capital:,.0f} | 실현손익: {realized_pnl:+,.0f} | 미실현손익: {unrealized_pnl:+,.0f} | 수익률: {total_return:+.2f}% | 보유포지션: {open_positions}개"
            send_log_to_dashboard(log_msg)
        
        # 일반 로그
        log_msg = f"[{timestamp}] {strategy} 전략 | {regime} 시장국면 | ML예측: {ml_prediction:+.2f}% | 현재가: 50000원"
        send_log_to_dashboard(log_msg)
        
        print(f"[{i+1}/30] 총자산: {current_capital:,.0f}원 | 수익률: {total_return:+.2f}% | 포지션: {open_positions}개")
        
        time.sleep(1)  # 1초 대기
    
    print("✅ 테스트 백테스트 완료!")
    print(f"📊 최종 결과: {current_capital:,.0f}원 ({total_return:+.2f}%)")

if __name__ == '__main__':
    simulate_backtest() 