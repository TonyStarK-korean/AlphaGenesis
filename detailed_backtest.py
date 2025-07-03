# detailed_backtest.py
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

class DetailedBacktestLogger:
    """상세 백테스트 로거"""
    
    def __init__(self):
        self.trade_count = 0
        self.total_profit = 0
        self.total_loss = 0
        self.win_count = 0
        self.loss_count = 0
        
    def print_trade_log(self, trade_info):
        """거래 로그 출력"""
        self.trade_count += 1
        
        # 수익/손실 계산
        pnl = trade_info['pnl']
        pnl_pct = trade_info['pnl_pct']
        
        if pnl > 0:
            self.total_profit += pnl
            self.win_count += 1
            result_emoji = "📈"
            result_text = "수익"
        else:
            self.total_loss += abs(pnl)
            self.loss_count += 1
            result_emoji = "📉"
            result_text = "손실"
        
        # 로그 출력
        print(f"{result_emoji} 거래 #{self.trade_count:03d} | "
              f"{trade_info['timestamp']} | "
              f"{trade_info['symbol']:8s} | "
              f"{trade_info['market_condition']:6s} | "
              f"{trade_info['direction']:4s} | "
              f"{trade_info['strategy']:12s} | "
              f"{pnl_pct:+6.2f}% | "
              f"{pnl:+9,.0f}원 | "
              f"{trade_info['remaining_capital']:,.0f}원")
        
        # 추가 정보 (진입/청산 가격)
        print(f"   진입: {trade_info['entry_price']:,.2f} | "
              f"청산: {trade_info['exit_price']:,.2f} | "
              f"레버리지: {trade_info['leverage']:.1f}배 | "
              f"사유: {trade_info['reason']}")
        print("-" * 120)
    
    def print_summary(self, initial_capital, final_capital):
        """최종 요약 출력"""
        total_return = (final_capital - initial_capital) / initial_capital * 100
        win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0
        
        print(f"\n{'='*120}")
        print(f"📊 백테스트 최종 결과")
        print(f"{'='*120}")
        print(f"💰 초기 자본: {initial_capital:,.0f}원")
        print(f"💰 최종 자본: {final_capital:,.0f}원")
        print(f"📈 총 수익률: {total_return:+.2f}%")
        print(f"📊 총 거래 수: {self.trade_count}건")
        print(f"✅ 승리 거래: {self.win_count}건")
        print(f"❌ 손실 거래: {self.loss_count}건")
        print(f"�� 승률: {win_rate:.1f}%")
        print(f"📈 총 수익: {self.total_profit:,.0f}원")
        print(f"📉 총 손실: {self.total_loss:,.0f}원")
        print(f"💵 순손익: {self.total_profit - self.total_loss:,.0f}원")
        print(f"{'='*120}")

def run_detailed_backtest_with_logs():
    """상세 로그와 함께 백테스트 실행"""
    
    print("�� AlphaGenesis 상세 백테스트 시작")
    print("=" * 120)
    
    # 설정
    initial_capital = 10_000_000  # 1000만원
    commission_rate = 0.0004      # 0.04% 수수료
    slippage_rate = 0.0002        # 0.02% 슬리피지
    
    # 1. 데이터 로드
    print("📥 데이터 로드 중...")
    try:
        # 여러 종목 데이터 로드
        symbols = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT', 'ADA_USDT', 'DOT_USDT']
        all_data = {}
        
        for symbol in symbols:
            file_path = f'data/market_data/{symbol}_1h.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                df['symbol'] = symbol.replace('_', '/')
                all_data[symbol] = df
                print(f"✅ {symbol}: {len(df)}개 캔들 로드")
            else:
                print(f"⚠️ {symbol} 데이터 파일 없음")
        
        if not all_data:
            print("❌ 데이터 파일을 찾을 수 없습니다.")
            return
        
        # 첫 번째 종목으로 백테스트 실행
        symbol = list(all_data.keys())[0]
        df = all_data[symbol]
        print(f"\n📊 백테스트 종목: {symbol}")
        print(f"📅 기간: {df.index[0]} ~ {df.index[-1]}")
        
    except Exception as e:
        print(f"❌ 데이터 로드 오류: {e}")
        return
    
    # 2. 피처 생성
    print("\n🔧 피처 생성 중...")
    df_features = make_features(df.copy())
    df_features = generate_crypto_features(df_features)
    df_features = generate_advanced_features(df_features)
    print(f"✅ 피처 생성 완료: {len(df_features.columns)}개 피처")
    
    # 3. ML 모델 훈련
    print("\n🤖 ML 모델 훈련 중...")
    model = PricePredictionModel()
    model.fit(df_features)
    print("✅ ML 모델 훈련 완료")
    
    # 4. 백테스트 실행
    print("\n�� 백테스트 실행 중...")
    print("=" * 120)
    print("종목      | 시장국면 | 방향 | 전략        | 수익률   | 수익금      | 남은자산")
    print("-" * 120)
    
    # 백테스트 변수 초기화
    current_capital = initial_capital
    position = 0  # 0: 중립, 1: 롱, -1: 숏
    entry_price = 0
    entry_time = None
    strategy_used = ""
    leverage = 1.0
    
    # 로거 초기화
    logger = DetailedBacktestLogger()
    
    # 시장 국면 분석
    market_condition = detect_market_condition_simple(df_features['close'].values)
    
    # 기본 파라미터
    params = {
        'confidence_threshold': 0.3,
        'leverage_multiplier': 1.0,
        'max_leverage': 5,
        'position_size_multiplier': 1.0,
        'base_position_size': 0.1,
        'stop_loss_multiplier': 1.0,
        'take_profit_multiplier': 1.0,
        'cvd_weight': 0.5,
        'multi_timeframe_weight': 0.5,
        'ml_prediction_weight': 0.7,
        'volatility_threshold': 0.1,
        'volume_threshold': 2.0,
        'asia_time_multiplier': 1.0,
        'europe_time_multiplier': 1.0,
        'us_time_multiplier': 1.0
    }
    
    # 백테스트 루프
    for i, (timestamp, row) in enumerate(df_features.iterrows()):
        try:
            # ML 예측
            ml_pred = model.predict(df_features.iloc[i:i+1])
            if isinstance(ml_pred, (list, np.ndarray)):
                ml_pred = ml_pred[0] if len(ml_pred) > 0 else 0
            
            # 거래 신호 생성 (기존 함수 사용)
            signal = generate_crypto_trading_signal(row, ml_pred, market_condition, params)
            
            # 포지션이 없을 때 진입 신호 확인
            if position == 0 and signal['signal'] != 0:
                # 진입
                position = signal['signal']  # 1: 롱, -1: 숏
                entry_price = row['close']
                entry_time = timestamp
                strategy_used = get_strategy_name(signal)
                leverage = signal.get('leverage_suggestion', 2.0)
                
                direction = "LONG" if position == 1 else "SHORT"
                print(f"📈 진입 | {row['symbol']:8s} | {market_condition:6s} | "
                      f"{direction:4s} | {strategy_used:12s} | "
                      f"진입가: {entry_price:,.2f} | 레버리지: {leverage:.1f}배")
            
            # 포지션이 있을 때 청산 조건 확인
            elif position != 0:
                current_price = row['close']
                
                # 손절/익절 확인
                should_close = False
                close_reason = ""
                
                if position == 1:  # 롱 포지션
                    stop_loss = signal.get('stop_loss', entry_price * 0.98)  # 기본 2% 손절
                    take_profit = signal.get('take_profit', entry_price * 1.05)  # 기본 5% 익절
                    
                    if current_price <= stop_loss:
                        should_close = True
                        close_reason = "손절"
                    elif current_price >= take_profit:
                        should_close = True
                        close_reason = "익절"
                        
                else:  # 숏 포지션
                    stop_loss = signal.get('stop_loss', entry_price * 1.02)  # 기본 2% 손절
                    take_profit = signal.get('take_profit', entry_price * 0.95)  # 기본 5% 익절
                    
                    if current_price >= stop_loss:
                        should_close = True
                        close_reason = "손절"
                    elif current_price <= take_profit:
                        should_close = True
                        close_reason = "익절"
                
                # 청산 실행
                if should_close:
                    # 수익/손실 계산
                    if position == 1:  # 롱
                        price_change = (current_price - entry_price) / entry_price
                    else:  # 숏
                        price_change = (entry_price - current_price) / entry_price
                    
                    # 레버리지 적용 및 수수료 차감
                    pnl_pct = price_change * leverage
                    pnl_amount = current_capital * pnl_pct * 0.1  # 10% 포지션 크기
                    
                    # 수수료 및 슬리피지 차감
                    fees = abs(pnl_amount) * (commission_rate + slippage_rate)
                    net_pnl = pnl_amount - fees
                    
                    # 자본 업데이트
                    current_capital += net_pnl
                    
                    # 거래 로그 생성
                    trade_info = {
                        'timestamp': entry_time.strftime('%Y-%m-%d %H:%M'),
                        'symbol': row['symbol'],
                        'market_condition': market_condition,
                        'direction': 'LONG' if position == 1 else 'SHORT',
                        'strategy': strategy_used,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'leverage': leverage,
                        'pnl': net_pnl,
                        'pnl_pct': pnl_pct * 100,
                        'reason': close_reason,
                        'remaining_capital': current_capital
                    }
                    
                    # 로그 출력
                    logger.print_trade_log(trade_info)
                    
                    # 포지션 리셋
                    position = 0
                    entry_price = 0
                    entry_time = None
                    strategy_used = ""
                    leverage = 1.0
            
            # 진행률 표시
            if i % 1000 == 0:
                progress = (i / len(df_features)) * 100
                print(f"진행률: {progress:.1f}% ({i}/{len(df_features)})")
                
        except Exception as e:
            if i % 1000 == 0:  # 1000회마다만 오류 출력
                print(f"⚠️ 백테스트 오류 (행 {i}): {e}")
            continue
    
    # 최종 결과 출력
    logger.print_summary(initial_capital, current_capital)
    
    return {
        'initial_capital': initial_capital,
        'final_capital': current_capital,
        'total_return': (current_capital - initial_capital) / initial_capital * 100,
        'total_trades': logger.trade_count,
        'win_rate': (logger.win_count / logger.trade_count * 100) if logger.trade_count > 0 else 0
    }

def get_strategy_name(signal):
    """신호에서 전략 이름 추출"""
    confidence = signal.get('confidence', 0)
    
    if confidence > 0.8:
        return "STRONG_SIGNAL"
    elif confidence > 0.6:
        return "MEDIUM_SIGNAL"
    elif confidence > 0.4:
        return "WEAK_SIGNAL"
    else:
        return "BASIC_SIGNAL"

if __name__ == "__main__":
    results = run_detailed_backtest_with_logs()
    
    if results:
        print(f"\n�� 백테스트 완료!")
        print(f"   최종 수익률: {results['total_return']:.2f}%")
        print(f"   총 거래 수: {results['total_trades']}건")
        print(f"   승률: {results['win_rate']:.1f}%")