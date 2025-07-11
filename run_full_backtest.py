#!/usr/bin/env python3
"""
🚀 AlphaGenesis: 통합 백테스트 실행기
데이터 다운로드, ML 모델 훈련, 전체 전략 백테스트를 한 번에 실행합니다.
"""

import sys
import os
import logging
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

# 프로젝트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- 필요한 모듈 임포트 ---
try:
    from local_data_downloader_fixed import LocalDataDownloaderFixed
    from run_ml_backtest import PricePredictionModel, make_features, generate_crypto_features, generate_advanced_features, setup_logging
    from triple_combo_strategy import TripleComboStrategy, check_position_exit, calculate_pnl
    from backtest_logger import BacktestLogger
    print("✅ 모든 필수 모듈 로드 성공!")
except ImportError as e:
    print(f"❌ 필수 모듈 로드 실패: {e}")
    print("   실행에 필요한 모든 .py 파일이 있는지 확인해주세요.")
    sys.exit(1)

warnings.filterwarnings('ignore')

class BacktestEngine:
    """
    다중 시간 프레임(MTF) 및 모든 전략 규칙을 통합한 백테스트 엔진.
    """
    def __init__(self, initial_capital, strategy_manager, model, logger):
        # 설정
        self.initial_capital = initial_capital
        self.strategy_manager = strategy_manager
        self.model = model
        self.logger = logger

        # 상태 변수
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.position = 0
        self.position_info = {}
        self.trades = []
        self.equity_curve = [{'time': None, 'capital': initial_capital}]
        
        # Phase 및 리스크 관리 변수
        self.phase = 'NORMAL' # NORMAL, AGGRESSIVE, DEFENSIVE
        self.consecutive_wins = 0
        self.consecutive_losses = 0

    def _update_phase(self, row):
        """
        거래 성과와 시장 상황에 따라 공격/방어 모드를 전환합니다.
        """
        volatility = row.get('volatility_20', 0.05)
        rsi = row.get('rsi_14', 50)
        
        # 방어 모드 진입 조건
        if self.consecutive_losses >= 3 or volatility > 0.08:
            if self.phase != 'DEFENSIVE':
                self.logger.log_system_event(row.name, f"🛡️ 방어 모드 전환 (연속 {self.consecutive_losses}패 / 변동성 {volatility:.2%})")
                self.phase = 'DEFENSIVE'
        
        # 공격 모드 진입 조건
        elif self.consecutive_wins >= 5 and volatility < 0.05 and rsi < 70:
            if self.phase != 'AGGRESSIVE':
                self.logger.log_system_event(row.name, f"⚔️ 공격 모드 전환 (연속 {self.consecutive_wins}승)")
                self.phase = 'AGGRESSIVE'
        
        # 일반 모드 복귀 조건
        else:
            if self.phase != 'NORMAL':
                self.logger.log_system_event(row.name, "😐 일반 모드로 복귀")
                self.phase = 'NORMAL'

    def _calculate_dynamic_leverage(self, signal, row):
        """
        문서의 모든 규칙에 따라 동적 레버리지를 계산합니다.
        """
        reasons = []
        market_regime = signal.get('market_phase', '횡보장')
        
        # 1. 시장 국면별 기본 레버리지
        regime_leverage_map = {'급등장': 2.5, '상승장': 2.0, '횡보장': 1.5, '하락장': 1.0, '급락장': 0.8}
        leverage = regime_leverage_map.get(market_regime, 1.5)
        reasons.append(f"기본({market_regime}):{leverage:.1f}x")

        # 2. ML 예측 기반 조정
        ml_pred = signal.get('ml_pred', 0)
        ml_adj = 1.0 + (ml_pred * 10) # ML 예측 1%당 레버리지 10% 조정
        leverage *= ml_adj
        reasons.append(f"ML({ml_pred:+.2%}):x{ml_adj:.2f}")

        # 3. Phase 기반 조정
        if self.phase == 'AGGRESSIVE':
            leverage *= 1.2
            reasons.append(f"Phase(공격):x1.2")
        elif self.phase == 'DEFENSIVE':
            leverage *= 0.7
            reasons.append(f"Phase(방어):x0.7")

        # 4. 낙폭 기반 조정
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital if self.peak_capital > 0 else 0
        if current_drawdown > 0.15:
            leverage *= 0.6 # 40% 감소
            reasons.append(f"낙폭({current_drawdown:.1%}):x0.6")

        # 5. 연속 손실 기반 조정
        if self.consecutive_losses >= 4:
            leverage *= 0.5 # 50% 감소
            reasons.append(f"연속손실({self.consecutive_losses}회):x0.5")

        # 6. 변동성 기반 조정
        volatility = row.get('volatility_20', 0.05)
        if volatility > 0.10:
            leverage *= 0.7 # 30% 감소
            reasons.append(f"변동성({volatility:.1%}):x0.7")

        final_leverage = np.clip(leverage, 0.5, 7.0) # 최종 레버리지 0.5x ~ 7.0x 범위
        return final_leverage, ", ".join(reasons)

    def _find_entry_trigger_5m(self, df_5m_slice: pd.DataFrame, direction: int):
        """
        5분봉에서 정밀 진입 시점을 찾습니다.
        (예시: 5분봉에서 단기 이평선이 중기 이평선을 돌파할 때 진입)
        """
        # 5분봉 데이터에 이동평균이 없으면 계산
        if 'ma_5' not in df_5m_slice.columns:
            df_5m_slice['ma_5'] = df_5m_slice['close'].rolling(5).mean()
            df_5m_slice['ma_20'] = df_5m_slice['close'].rolling(20).mean()
            df_5m_slice.fillna(method='bfill', inplace=True)

        for _, row_5m in df_5m_slice.iterrows():
            ma_5 = row_5m.get('ma_5', 0)
            ma_20 = row_5m.get('ma_20', 0)
            if direction == 1 and ma_5 > ma_20:
                return row_5m
            elif direction == -1 and ma_5 < ma_20:
                return row_5m
        return None  # 진입 시점 못찾음

    def run(self, df_1h: pd.DataFrame, df_5m: pd.DataFrame):
        """
        다중 시간 프레임(MTF) 백테스트 실행
        - 1시간봉: 거시적 전략 결정
        - 5분봉: 정밀 진입/청산 타이밍
        """
        # --- 1. 피처 엔지니어링 ---
        print("🔧 1h 및 5m 데이터에 대한 고급 피처 생성 중...")
        df_1h_features = generate_advanced_features(generate_crypto_features(make_features(df_1h.copy())))
        df_1h_features.dropna(inplace=True)
        df_5m_features = make_features(df_5m.copy()) # 5분봉은 기본 피처만 사용
        df_5m_features.dropna(inplace=True)
        
        print("🤖 ML 예측 생성 중 (1h 기반)...")
        if self.model and hasattr(self.model, 'is_fitted') and self.model.is_fitted:
            df_1h_features['ml_prediction'] = self.model.predict(df_1h_features)
        else:
            df_1h_features['ml_prediction'] = 0.0
        
        if not df_1h_features.empty:
            self.equity_curve[0]['time'] = df_1h_features.index[0]

        print(f"\n📈 MTF 백테스트 실행 중 (총 {len(df_1h_features)}개 1h 캔들)...")
        
        # 1시간봉을 기준으로 메인 루프 실행
        for idx_1h, row_1h in tqdm(df_1h_features.iterrows(), total=len(df_1h_features), desc="AlphaGenesis MTF Backtest"):
            
            # --- 5분봉 단위로 정밀 청산/진입 확인 ---
            start_time_5m = idx_1h
            end_time_5m = idx_1h + timedelta(hours=1) - timedelta(seconds=1)
            df_5m_slice = df_5m_features[(df_5m_features.index >= start_time_5m) & (df_5m_features.index <= end_time_5m)]

            position_closed_in_hour = False

            # 해당 시간의 5분봉 캔들을 순회
            if self.position != 0:
                for _, row_5m in df_5m_slice.iterrows():
                    current_time = row_5m.name
                    current_price = row_5m['close']

                    # 1. 포지션 청산 확인 (5분봉 기준)
                    should_close, close_reason = check_position_exit(
                        row_5m, self.position, self.position_info['entry_price'], 
                        self.position_info['stop_loss'], self.position_info['take_profit']
                    )
                    if should_close:
                        pnl = calculate_pnl(self.position, self)
                        self.capital += pnl
                        self.peak_capital = max(self.peak_capital, self.capital)
                        self.trades.append({
                            'entry_time': self.position_info['entry_time'],
                            'exit_time': current_time,
                            'position': self.position,
                            'entry_price': self.position_info['entry_price'],
                            'exit_price': current_price,
                            'pnl': pnl,
                            'leverage': self.position_info.get('leverage', 1),
                            'reason': close_reason
                        })
                        self.position = 0
                        self.position_info = {}
                        position_closed_in_hour = True
                        self.consecutive_wins = 0
                        self.consecutive_losses += 1
                        self.logger.log_trade_event(current_time, 'CLOSE', pnl, self.capital)

            # --- 포지션 유지/진입 결정 ---
            if self.position == 0:
                # 포지션이 없을 때만 진입 신호 확인
                if row_1h.get('long_entry_signal') == 1:
                    # 롱 진입 신호
                    entry_price = row_1h['close']
                    stop_loss = entry_price * 0.98 # 진입가 대비 2% 손절
                    take_profit = entry_price * 1.05 # 진입가 대비 5% 익절
                    signal_strength = row_1h.get('signal_strength', 1)
                    dynamic_leverage, reasons = self._calculate_dynamic_leverage(row_1h, row_1h)
                    self.position = 1
                    self.position_info = {
                        'entry_time': idx_1h,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'leverage': dynamic_leverage,
                        'signal_strength': signal_strength,
                        'reasons': reasons
                    }
                    self.capital -= (entry_price * dynamic_leverage)
                    self.logger.log_trade_event(idx_1h, 'OPEN_LONG', entry_price, self.capital, dynamic_leverage, reasons)
                
                elif row_1h.get('short_entry_signal') == 1:
                    # 숏 진입 신호
                    entry_price = row_1h['close']
                    stop_loss = entry_price * 1.02 # 진입가 대비 2% 손절
                    take_profit = entry_price * 0.95 # 진입가 대비 5% 익절
                    signal_strength = row_1h.get('signal_strength', 1)
                    dynamic_leverage, reasons = self._calculate_dynamic_leverage(row_1h, row_1h)
                    self.position = -1
                    self.position_info = {
                        'entry_time': idx_1h,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'leverage': dynamic_leverage,
                        'signal_strength': signal_strength,
                        'reasons': reasons
                    }
                    self.capital += (entry_price * dynamic_leverage)
                    self.logger.log_trade_event(idx_1h, 'OPEN_SHORT', entry_price, self.capital, dynamic_leverage, reasons)

            # --- 포지션 상태에 따른 자본 곡선 업데이트 ---
            self.equity_curve.append({'time': idx_1h, 'capital': self.capital})

            # --- Phase 업데이트 ---
            self._update_phase(row_1h)

        print("✅ 백테스트 완료!")
        return pd.DataFrame(self.equity_curve).set_index('time')

def parse_arguments():
    """
    커맨드 라인 인자를 파싱합니다.
    """
    parser = argparse.ArgumentParser(description="AlphaGenesis 통합 백테스트 실행기")
    parser.add_argument("--start_date", type=str, default="2022-01-01", help="백테스트 시작 날짜 (형식: YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=str(datetime.now().date()), help="백테스트 종료 날짜 (형식: YYYY-MM-DD)")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="거래할 심볼")
    parser.add_argument("--interval", type=str, default="1h", help="캔들스틱 인터벌 (예: 1m, 5m, 1h, 1d)")
    parser.add_argument("--leverage", type=float, default=1.0, help="기본 레버리지 설정")
    parser.add_argument("--capital", type=float, default=100000, help="초기 자본금")
    parser.add_argument("--mode", type=str, default="test", help="모드 선택 (test 또는 live)")
    return parser.parse_args()

def main():
    """
    메인 함수: 전체 백테스트 프로세스를 관장합니다.
    """
    args = parse_arguments()

    # --- 0. 로깅 및 설정 ---
    log_filename = f"backtest_log_{args.symbol.replace('/', '_')}_{args.start_date}_to_{args.end_date}.txt"
    setup_logging(log_filename)
    logger = BacktestLogger(log_filename)
    logger.log_system_event(datetime.now(), "백테스트 시작", {
        "심볼": args.symbol,
        "시작일자": args.start_date,
        "종료일자": args.end_date,
        "기본레버리지": args.leverage,
        "초기자본금": args.capital,
        "모드": args.mode
    })

    # --- 1. 데이터 다운로드 및 전처리 ---
    print("📥 데이터 다운로드 및 전처리 중...")
    downloader = LocalDataDownloaderFixed(args.symbol, args.start_date, args.end_date, args.interval)
    df_1h, df_5m = downloader.get_data()
    
    if df_1h is None or df_5m is None:
        print("❌ 데이터프레임이 비어있습니다. 데이터 다운로드에 실패했습니다.")
        return

    # --- 2. 전략 및 모델 초기화 ---
    print("🛠️ 전략 및 모델 초기화 중...")
    strategy_manager = None # 전략 매니저 초기화 (필요시 추가 기능 구현)
    model = PricePredictionModel() # 기본 모델 인스턴스 생성 (추후 개선 가능)
    backtest_engine = BacktestEngine(args.capital, strategy_manager, model, logger)

    # --- 3. 백테스트 실행 ---
    result_equity_curve = backtest_engine.run(df_1h, df_5m)

    # --- 4. 결과 분석 및 저장 ---
    result_filename = f"backtest_result_{args.symbol.replace('/', '_')}_{args.start_date}_to_{args.end_date}.csv"
    result_equity_curve.to_csv(result_filename)
    print(f"📈 결과 저장 완료: {result_filename}")

    logger.log_system_event(datetime.now(), "백테스트 종료", {
        "최종자본금": backtest_engine.capital,
        "거래건수": len(backtest_engine.trades),
        "연속승리": backtest_engine.consecutive_wins,
        "연속패배": backtest_engine.consecutive_losses
    })

if __name__ == "__main__":
    main()
