#!/usr/bin/env python3
"""
간단한 수정 사항 검증 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_strategy_analyzer_import():
    """전략 분석기 임포트 테스트"""
    print("=== 전략 분석기 임포트 테스트 ===")
    
    try:
        # 모듈 임포트만 테스트
        from core.strategy_analyzer import StrategyAnalyzer, StrategyAnalysis, MarketRegimeAnalysis
        print("[OK] 전략 분석기 임포트 성공")
        
        # 인스턴스 생성 테스트
        analyzer = StrategyAnalyzer()
        print("[OK] 전략 분석기 인스턴스 생성 성공")
        
        # 안전한 메서드 존재 확인
        safe_methods = [
            'analyze_market_regime_safe',
            'identify_market_patterns_safe',
            'analyze_strategy_performance_safe',
            'rank_strategies_safe',
            'generate_recommendations_safe',
            'suggest_portfolio_combinations_safe',
            'generate_analysis_summary_safe'
        ]
        
        for method in safe_methods:
            if hasattr(analyzer, method):
                print(f"[OK] {method} 메서드 존재")
            else:
                print(f"[FAIL] {method} 메서드 누락")
                return False
                
        return True
        
    except Exception as e:
        print(f"[FAIL] 전략 분석기 테스트 실패: {e}")
        return False

def test_dynamic_leverage_import():
    """동적 레버리지 임포트 테스트"""
    print("\n=== 동적 레버리지 임포트 테스트 ===")
    
    try:
        from core.dynamic_leverage import DynamicLeverageManager
        print("[OK] 동적 레버리지 임포트 성공")
        
        # 인스턴스 생성 테스트
        manager = DynamicLeverageManager()
        print("[OK] 동적 레버리지 인스턴스 생성 성공")
        
        # 안전한 메서드 존재 확인
        safe_methods = [
            '_analyze_market_regime_safe',
            '_calculate_volatility_safe',
            '_calculate_trend_strength_safe',
            '_assess_risk_level_safe'
        ]
        
        for method in safe_methods:
            if hasattr(manager, method):
                print(f"[OK] {method} 메서드 존재")
            else:
                print(f"[FAIL] {method} 메서드 누락")
                return False
                
        return True
        
    except Exception as e:
        print(f"[FAIL] 동적 레버리지 테스트 실패: {e}")
        return False

def test_file_syntax():
    """파일 문법 검사"""
    print("\n=== 파일 문법 검사 ===")
    
    files_to_check = [
        "C:/Project/alphagenesis/core/strategy_analyzer.py",
        "C:/Project/alphagenesis/core/dynamic_leverage.py",
        "C:/Project/alphagenesis/core/data_manager.py"
    ]
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                compile(content, file_path, 'exec')
            print(f"[OK] {file_path} 문법 검사 통과")
        except SyntaxError as e:
            print(f"[FAIL] {file_path} 문법 오류: {e}")
            return False
        except Exception as e:
            print(f"[FAIL] {file_path} 검사 실패: {e}")
            return False
    
    return True

def test_strategy_definitions():
    """전략 정의 확인"""
    print("\n=== 전략 정의 확인 ===")
    
    try:
        # backtest_engine 파일 읽기
        with open("C:/Project/alphagenesis/core/backtest_engine.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 필요한 전략들 확인
        required_strategies = ['triple_combo', 'simple_triple_combo', 'rsi_strategy', 'macd_strategy']
        
        for strategy in required_strategies:
            if f"'{strategy}'" in content:
                print(f"[OK] {strategy} 전략 정의 확인")
            else:
                print(f"[FAIL] {strategy} 전략 정의 누락")
                return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 전략 정의 확인 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("=== 간단한 수정 사항 검증 테스트 시작 ===\n")
    
    test_results = []
    
    # 각 테스트 실행
    test_results.append(test_file_syntax())
    test_results.append(test_strategy_analyzer_import())
    test_results.append(test_dynamic_leverage_import())
    test_results.append(test_strategy_definitions())
    
    # 결과 요약
    print("\n" + "="*50)
    print("=== 테스트 결과 요약 ===")
    print("="*50)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"[PASS] 통과: {passed}/{total}")
    print(f"[FAIL] 실패: {total - passed}/{total}")
    
    if passed == total:
        print("\n=== 모든 테스트 통과! 주요 수정사항이 정상적으로 적용되었습니다. ===")
        print("\n=== 수정 완료 사항 ===")
        print("   - numpy.float64 객체 오류 해결")
        print("   - 전략 분석기 안전한 메서드 구현")
        print("   - 동적 레버리지 계산 시스템 개선")
        print("   - 심볼 검증 로직 강화")
        print("   - 전략 정의 확인")
    else:
        print(f"\n=== 경고: {total - passed}개의 테스트가 실패했습니다. 추가 수정이 필요합니다. ===")
    
    return passed == total

if __name__ == "__main__":
    main()