#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 실시간 백테스트 시뮬레이터 (대시보드 연동 테스트용)
패키지 의존성 최소화
"""

import requests
import time
import random
from datetime import datetime
import json

def send_log_to_dashboard(log_msg):
    """대시보드로 로그 전송"""
    urls = [
        'http://localhost:5000/api/realtime_log',
        'http://34.47.77.230:5000/api/realtime_log'
    ]
    
    success = False
    for url in urls:
        try:
            response = requests.post(
                url, 
                json={'log': log_msg}, 
                timeout=3
            )
            if response.status_code == 200:
                print(f"✅ 대시보드 전송 성공: {url}")
                success = True
                break
        except Exception as e:
            print(f"❌ 대시보드 전송 실패 ({url}): {e}")
            continue
    
    return success

def simulate_backtest(duration_seconds=60):
    """간단한 백테스트 시뮬레이션"""
    print(f"🎯 {duration_seconds}초간 실시간 백테스트 시뮬레이션 시작!")
    print("📊 대시보드 연동 테스트 중...")
    
    # 초기 설정
    initial_capital = 10000000  # 1천만원
    current_capital = initial_capital
    realized_pnl = 0
    unrealized_pnl = 0
    open_positions = 0
    trades_count = 0
    
    # 전략 및 시장국면 목록
    strategies = ['추세추종', '역추세', '모멘텀돌파', '숏모멘텀', '비트코인숏전략']
    regimes = ['급등', '상승', '횡보', '하락', '급락']
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < duration_seconds:
        iteration += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 자산 변동 시뮬레이션 (-2% ~ +3%)
        price_change = random.uniform(-0.02, 0.03)
        current_capital = max(current_capital * (1 + price_change), 1000000)
        
        total_pnl = current_capital - initial_capital
        total_return = (total_pnl / initial_capital) * 100
        
        # ML 예측값 (-20% ~ +20%)
        ml_prediction = random.uniform(-20, 20)
        
        # 랜덤 전략 및 시장국면
        strategy = random.choice(strategies)
        regime = random.choice(regimes)
        
        # 거래 발생 시뮬레이션 (30% 확률)
        if random.random() < 0.3:
            if random.random() < 0.6:  # 진입
                trades_count += 1
                open_positions = min(open_positions + 1, 5)
                action = "진입"
                direction = "매수" if random.random() < 0.5 else "매도"
                
                # 거래 로그 전송
                trade_log = f"[{timestamp}] | {action:^4} | {regime:^4} | {strategy:^10} | {direction:^4} | BTC/USDT | 50000.00 | 51000.00 | +2.00% | +50000 | {current_capital:>10,.0f} | 10.0% | 2.50배 | {strategy} 조건충족 | ML예측: {ml_prediction:.2f}%"
                send_log_to_dashboard(trade_log)
                
            else:  # 청산
                if open_positions > 0:
                    open_positions -= 1
                    action = "청산"
                    direction = "매수" if random.random() < 0.5 else "매도"
                    profit = random.uniform(-100000, 200000)
                    realized_pnl += profit
                    
                    # 청산 로그 전송
                    trade_log = f"[{timestamp}] | {action:^4} | {regime:^4} | {strategy:^10} | {direction:^4} | BTC/USDT | 51000.00 | 50500.00 | -1.00% | {profit:+,.0f} | {current_capital:>10,.0f} | 8.0% | 2.00배 | 수익실현 | ML예측: {ml_prediction:.2f}%"
                    send_log_to_dashboard(trade_log)
        
        # 매매 현황 로그 (5초마다 또는 포지션 보유시)
        if iteration % 5 == 0 or open_positions > 0:
            unrealized_pnl = random.uniform(-200000, 300000) if open_positions > 0 else 0
            status_log = f"[{timestamp}] === 매매 현황 === | 총자산: {current_capital:,.0f} | 실현손익: {realized_pnl:+,.0f} | 미실현손익: {unrealized_pnl:+,.0f} | 수익률: {total_return:+.2f}% | 보유포지션: {open_positions}개"
            send_log_to_dashboard(status_log)
        
        # 일반 전략 로그
        strategy_log = f"[{timestamp}] {strategy} 전략 | {regime} 시장국면 | ML예측: {ml_prediction:+.2f}% | 현재가: 50000원"
        send_log_to_dashboard(strategy_log)
        
        # 콘솔 출력
        elapsed = int(time.time() - start_time)
        print(f"[{elapsed:02d}초] 총자산: {current_capital:,.0f}원 | 수익률: {total_return:+.2f}% | 포지션: {open_positions}개 | 거래: {trades_count}회")
        
        time.sleep(1)  # 1초 대기
    
    # 최종 결과
    final_return = (current_capital - initial_capital) / initial_capital * 100
    final_log = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🎉 백테스트 완료! 최종 수익률: {final_return:+.2f}% | 총 거래: {trades_count}회"
    send_log_to_dashboard(final_log)
    
    print("=" * 60)
    print("✅ 실시간 백테스트 시뮬레이션 완료!")
    print(f"📊 최종 결과:")
    print(f"   초기자본: {initial_capital:,}원")
    print(f"   최종자본: {current_capital:,.0f}원")
    print(f"   총 수익률: {final_return:+.2f}%")
    print(f"   총 거래수: {trades_count}회")
    print(f"   실현손익: {realized_pnl:+,.0f}원")
    print("=" * 60)
    print("🌐 브라우저에서 http://localhost:5000 에서 실시간 결과를 확인하세요!")

def main():
    """메인 실행 함수"""
    print("🎯 실시간 대시보드 연동 테스트")
    print("=" * 50)
    
    try:
        # 연결 테스트
        print("🔍 대시보드 연결 테스트 중...")
        test_success = False
        
        for url in ['http://localhost:5000', 'http://34.47.77.230:5000']:
            try:
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    print(f"✅ 대시보드 연결 성공: {url}")
                    test_success = True
                    break
            except:
                print(f"❌ 대시보드 연결 실패: {url}")
        
        if not test_success:
            print("❌ 대시보드에 연결할 수 없습니다.")
            print("먼저 대시보드를 실행하세요: python dashboard/simple_dashboard.py")
            return
        
        # 시뮬레이션 시간 선택
        print("\n⏰ 시뮬레이션 시간을 선택하세요:")
        print("1. 30초 (빠른 테스트)")
        print("2. 60초 (기본)")
        print("3. 120초 (상세 테스트)")
        
        choice = input("선택 (1-3, 기본값: 2): ").strip()
        
        duration_map = {'1': 30, '2': 60, '3': 120}
        duration = duration_map.get(choice, 60)
        
        print(f"\n🚀 {duration}초간 시뮬레이션을 시작합니다...")
        time.sleep(2)
        
        simulate_backtest(duration)
        
    except KeyboardInterrupt:
        print("\n👋 사용자가 중단했습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == '__main__':
    main() 