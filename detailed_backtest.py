# improved_detailed_backtest.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트를 파이썬 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 기존 모듈들 임포트
from run_ml_backtest import (
    make_features, 
    generate_crypto_features, 
    generate_advanced_features, 
    detect_market_condition_simple,
    generate_crypto_trading_signal
)
from ml.models.price_prediction_model import PricePredictionModel

class ImprovedBacktestLogger:
    """개선된 백테스트 로거"""
    
    def __init__(self):
        self.trade_count = 0
        self.total_profit = 0
        self.total_loss = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.initial_capital = 10_000_000
        self.current_capital = self.initial_capital
        self.trades = []
        
    def log_entry(self, symbol, market_condition, direction, strategy, entry_price, leverage, position_size_pct):
        """진입 로그"""
        print(f"📈 진입 | {symbol} | {market_condition} | {direction} | {strategy} | "
              f"진입가: {entry_price:,.2f} | 레버리지: {leverage}배 | 진입비중: {position_size_pct:.1f}%")
    
    def log_exit(self, trade_num, exit_time, symbol, market_condition, direction, strategy, 
                 profit_pct, profit_amount, exit_price, entry_price, leverage, reason):
        """청산 로그"""
        profit_symbol = "📈" if profit_amount >= 0 else "📉"
        print(f"{profit_symbol} 거래 #{trade_num:03d} | {exit_time} | {symbol} | {market_condition} | "
              f"{direction} | {strategy} | {profit_pct:+.2f}% | {profit_amount:+,8.0f}원 | "
              f"{self.current_capital:,.0f}원")
        print(f"   진입: {entry_price:,.2f} | 청산: {exit_price:,.2f} | 레버리지: {leverage}배 | 사유: {reason}")
        print("-" * 120)
        
        # 통계 업데이트
        if profit_amount >= 0:
            self.win_trades += 1
            self.total_profit += profit_amount
        else:
            self.loss_trades += 1
            self.total_loss += abs(profit_amount)
            
        self.trades.append({
            'trade_num': trade_num,
            'symbol': symbol,
            'direction': direction,
            'strategy': strategy,
            'profit_pct': profit_pct,
            'profit_amount': profit_amount,
            'leverage': leverage,
            'reason': reason
        })
    
    def print_final_results(self):
        """최종 결과 출력"""
        total_trades = self.win_trades + self.loss_trades
        win_rate = (self.win_trades / total_trades * 100) if total_trades > 0 else 0
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital * 100)
        
        print("=" * 120)
        print("📊 백테스트 최종 결과")
        print("=" * 120)
        print(f"💰 초기 자본: {self.initial_capital:,}원")
        print(f"💰 최종 자본: {self.current_capital:,}원")
        print(f"📈 총 수익률: {total_return:+.2f}%")
        print(f"📊 총 거래 수: {total_trades}건")
        print(f"✅ 승리 거래: {self.win_trades}건")
        print(f"❌ 손실 거래: {self.loss_trades}건")
        print(f" 승률: {win_rate:.1f}%")
        print(f"📈 총 수익: {self.total_profit:,}원")
        print(f"📉 총 손실: {self.total_loss:,}원")
        print(f"💵 순손익: {self.total_profit - self.total_loss:+,}원")
        print("=" * 120)

def improved_backtest():
    """개선된 백테스트 실행"""
    print(" 트리플 콤보 전략 모듈 로드 성공!")
    print("AlphaGenesis 개선된 상세 백테스트 시작")
    print("=" * 120)
    
    # 데이터 로드
    print("📥 데이터 로드 중...")
    data_path = "data/market_data/"
    symbols = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT', 'ADA_USDT', 'DOT_USDT']
    
    all_data = {}
    for symbol in symbols:
        try:
            df = pd.read_csv(f"{data_path}{symbol}_1h.csv")
            print(f"✅ {symbol}: {len(df)}개 캔들 로드")
            all_data[symbol] = df
        except FileNotFoundError:
            print(f"❌ {symbol}: 파일 없음")
            continue
    
    if not all_data:
        print("❌ 데이터를 찾을 수 없습니다!")
        return
    
    # 백테스트 설정
    target_symbol = 'BTC_USDT'
    df = all_data[target_symbol].copy()
    
    print(f"\n📊 백테스트 종목: {target_symbol}")
    print(f"📅 기간: {df.iloc[0]['timestamp']} ~ {df.iloc[-1]['timestamp']}")
    
    # 피처 생성
    print("\n🔧 피처 생성 중...")
    df = make_features(df)
    print("✅ 피처 생성 완료")
    
    # ML 모델 훈련
    print("\n🤖 ML 모델 훈련 중...")
    model = PricePredictionModel()
    model.train(df)
    print("✅ ML 모델 훈련 완료")
    
    # 백테스트 실행
    print("\n백테스트 실행 중...")
    print("=" * 120)
    print("종목      | 시장국면 | 방향 | 전략        | 수익률   | 수익금      | 남은자산")
    print("-" * 120)
    
    logger = ImprovedBacktestLogger()
    
    # 백테스트 로직 (개선된 버전)
    for i in range(100, len(df) - 1):  # 100개 캔들 이후부터 시작
        try:
            current_data = df.iloc[:i+1]
            next_data = df.iloc[i+1]
            
            # 시장 상황 감지
            market_condition = detect_market_condition_simple(current_data)
            
            # 신호 생성 (개선된 로직)
            signal = generate_crypto_trading_signal(current_data, model)
            
            if signal['signal'] != 'HOLD':
                # 진입 비중 계산 (개선된 로직)
                confidence = abs(signal['confidence'])
                position_size_pct = min(confidence * 10, 20)  # 최대 20%
                
                # 레버리지 계산 (개선된 로직)
                if confidence > 0.8:
                    leverage = 2.0
                    strategy = "STRONG_SIGNAL"
                elif confidence > 0.6:
                    leverage = 1.5
                    strategy = "MEDIUM_SIGNAL"
                else:
                    leverage = 1.2
                    strategy = "WEAK_SIGNAL"
                
                # 진입 로그
                logger.log_entry(
                    symbol=f"{target_symbol.replace('_', '/')}",
                    market_condition=market_condition,
                    direction=signal['signal'],
                    strategy=strategy,
                    entry_price=current_data.iloc[-1]['close'],
                    leverage=leverage,
                    position_size_pct=position_size_pct
                )
                
                # 청산 로직 (개선된 로직)
                entry_price = current_data.iloc[-1]['close']
                position_value = logger.current_capital * (position_size_pct / 100) * leverage
                
                # 손익 계산
                if signal['signal'] == 'LONG':
                    profit_pct = (next_data['close'] - entry_price) / entry_price * leverage * 100
                else:  # SHORT
                    profit_pct = (entry_price - next_data['close']) / entry_price * leverage * 100
                
                profit_amount = position_value * (profit_pct / 100)
                logger.current_capital += profit_amount
                logger.trade_count += 1
                
                # 청산 사유 결정
                if profit_pct >= 1.0:
                    reason = "익절"
                elif profit_pct <= -2.0:
                    reason = "손절"
                else:
                    reason = "시간청산"
                
                # 청산 로그
                logger.log_exit(
                    trade_num=logger.trade_count,
                    exit_time=next_data['timestamp'],
                    symbol=f"{target_symbol.replace('_', '/')}",
                    market_condition=market_condition,
                    direction=signal['signal'],
                    strategy=strategy,
                    profit_pct=profit_pct,
                    profit_amount=profit_amount,
                    exit_price=next_data['close'],
                    entry_price=entry_price,
                    leverage=leverage,
                    reason=reason
                )
                
        except Exception as e:
            print(f"⚠️ 백테스트 오류 (행 {i}): {str(e)}")
            continue
    
    # 최종 결과 출력
    logger.print_final_results()
    print(f"\n백테스트 완료!")
    print(f"   최종 수익률: {((logger.current_capital - logger.initial_capital) / logger.initial_capital * 100):+.2f}%")
    print(f"   총 거래 수: {logger.trade_count}건")
    print(f"   승률: {(logger.win_trades / logger.trade_count * 100):.1f}%" if logger.trade_count > 0 else "   승률: 0%")

if __name__ == "__main__":
    improved_backtest()