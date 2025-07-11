#!/usr/bin/env python3
"""
백테스트 시스템 수정 사항 테스트
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from core.backtest_engine import RealBacktestEngine
from core.strategy_analyzer import StrategyAnalyzer
from core.data_manager import DataManager
from core.dynamic_leverage import DynamicLeverageManager
import pandas as pd

async def test_strategy_analyzer():
    """전략 분석기 테스트"""
    print("=== 전략 분석기 테스트 ===")
    
    try:
        analyzer = StrategyAnalyzer()
        
        # 테스트 날짜 설정
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # 시장 국면 분석 테스트
        print("1. 시장 국면 분석 테스트...")
        market_regime = await analyzer.analyze_market_regime_safe(start_date, end_date)
        print(f"   시장 국면: {market_regime.regime_type}")
        print(f"   변동성 수준: {market_regime.volatility_level}")
        print(f"   트렌드 강도: {market_regime.trend_strength}")
        
        print("✅ 전략 분석기 테스트 통과")
        return True
        
    except Exception as e:
        print(f"❌ 전략 분석기 테스트 실패: {e}")
        return False

async def test_data_manager():
    """데이터 매니저 테스트"""
    print("\n=== 데이터 매니저 테스트 ===")
    
    try:
        data_manager = DataManager()
        
        # 심볼 검증 테스트
        print("1. 심볼 검증 테스트...")
        valid_symbols = ['BTC/USDT', 'ETH/USDT', 'SHIB/USDT', 'COTI/USDT']
        invalid_symbols = ['INVALID/USDT', 'NOTEXIST/USDT']
        
        for symbol in valid_symbols:
            if not data_manager._is_valid_symbol(symbol):
                print(f"   ❌ {symbol} 검증 실패")
                return False
            else:
                print(f"   ✅ {symbol} 검증 통과")
        
        for symbol in invalid_symbols:
            if data_manager._is_valid_symbol(symbol):
                print(f"   ❌ {symbol} 잘못된 검증 통과")
                return False
            else:
                print(f"   ✅ {symbol} 정상적으로 거부")
        
        print("✅ 데이터 매니저 테스트 통과")
        return True
        
    except Exception as e:
        print(f"❌ 데이터 매니저 테스트 실패: {e}")
        return False

def test_dynamic_leverage():
    """동적 레버리지 테스트"""
    print("\n=== 동적 레버리지 테스트 ===")
    
    try:
        leverage_manager = DynamicLeverageManager()
        
        # 테스트 데이터 생성
        print("1. 테스트 데이터 생성...")
        test_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [99, 98, 97, 96, 95],
            'close': [104, 103, 102, 101, 100],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # 레버리지 계산 테스트
        print("2. 레버리지 계산 테스트...")
        result = leverage_manager.calculate_optimal_leverage(
            market_data=test_data,
            strategy='triple_combo',
            current_position=0.0,
            portfolio_value=100000.0
        )
        
        print(f"   최적 레버리지: {result.get('optimal_leverage', 'N/A')}")
        print(f"   시장 국면: {result.get('market_regime', 'N/A')}")
        print(f"   변동성: {result.get('volatility', 'N/A')}")
        
        # 안전한 메서드 테스트
        print("3. 안전한 메서드 테스트...")
        market_regime = leverage_manager._analyze_market_regime_safe(test_data)
        volatility = leverage_manager._calculate_volatility_safe(test_data)
        trend_strength = leverage_manager._calculate_trend_strength_safe(test_data)
        
        print(f"   시장 국면: {market_regime}")
        print(f"   변동성: {volatility}")
        print(f"   트렌드 강도: {trend_strength}")
        
        print("✅ 동적 레버리지 테스트 통과")
        return True
        
    except Exception as e:
        print(f"❌ 동적 레버리지 테스트 실패: {e}")
        return False

async def test_backtest_engine():
    """백테스트 엔진 테스트"""
    print("\n=== 백테스트 엔진 테스트 ===")
    
    try:
        engine = RealBacktestEngine()
        
        # 지원 전략 확인
        print("1. 지원 전략 확인...")
        required_strategies = ['triple_combo', 'simple_triple_combo', 'rsi_strategy', 'macd_strategy']
        
        for strategy in required_strategies:
            if strategy not in engine.strategies:
                print(f"   ❌ {strategy} 전략 누락")
                return False
            else:
                print(f"   ✅ {strategy} 전략 확인")
        
        # 레버리지 통계 계산 테스트
        print("2. 레버리지 통계 계산 테스트...")
        test_leverage_history = [1.0, 1.5, 2.0, {'optimal_leverage': 2.5}, 1.8]
        leverage_stats = engine.calculate_leverage_stats(test_leverage_history)
        
        print(f"   평균 레버리지: {leverage_stats.get('avg', 'N/A')}")
        print(f"   최대 레버리지: {leverage_stats.get('max', 'N/A')}")
        print(f"   최소 레버리지: {leverage_stats.get('min', 'N/A')}")
        
        print("✅ 백테스트 엔진 테스트 통과")
        return True
        
    except Exception as e:
        print(f"❌ 백테스트 엔진 테스트 실패: {e}")
        return False

async def main():
    """메인 테스트 실행"""
    print("🧪 백테스트 시스템 수정 사항 테스트 시작\n")
    
    test_results = []
    
    # 각 테스트 실행
    test_results.append(await test_strategy_analyzer())
    test_results.append(await test_data_manager())
    test_results.append(test_dynamic_leverage())
    test_results.append(await test_backtest_engine())
    
    # 결과 요약
    print("\n" + "="*50)
    print("🏁 테스트 결과 요약")
    print("="*50)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ 통과: {passed}/{total}")
    print(f"❌ 실패: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과! 시스템이 정상적으로 수정되었습니다.")
    else:
        print(f"\n⚠️  {total - passed}개의 테스트가 실패했습니다. 추가 수정이 필요합니다.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())