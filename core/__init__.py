"""
AlphaGenesis Core Module
중앙 집중식 import 관리 및 순환 참조 방지
"""

# 설정 모듈 (최우선)
try:
    from ..config.unified_config import config, UnifiedConfig
except ImportError:
    # 상대 import 실패시 절대 import 시도
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config.unified_config import config, UnifiedConfig

# 기본 모듈들 (순환 참조 방지 순서)
try:
    # 1. 예외 처리 (다른 모듈에서 사용)
    from .exceptions import *
    
    # 2. 데이터 관리 (기반 모듈)
    from .data_manager import DataManager
    
    # 3. 위험 관리 (독립적)
    from .risk_management import RiskManager
    
    # 4. 동적 레버리지 (위험 관리 의존)
    from .dynamic_leverage import DynamicLeverageManager
    
    # 5. ML 최적화 (데이터 관리 의존)
    from .ml_optimizer import MLOptimizer
    
    # 6. 포지션 관리 (위험 관리 의존)
    from .position_management import PositionManager
    
    # 7. 백테스트 엔진 (위 모듈들 의존)
    from .backtest_engine import RealBacktestEngine, BacktestResult
    
    # 8. 전략 분석 (백테스트 엔진 의존)
    from .strategy_analyzer import StrategyAnalyzer
    
    # 9. 포트폴리오 최적화 (전략 분석 의존)
    from .portfolio_optimizer import PortfolioOptimizer
    
    # 10. 실시간 거래 엔진 (모든 모듈 의존)
    from .live_trading_engine import LiveTradingEngine
    
    # 11. 레거시 거래 엔진들 (하위 호환성)
    from .trading_engine import *
    
except ImportError as e:
    # 개별 모듈 import 실패시 로깅
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"일부 모듈 import 실패: {e}")

# 버전 정보
__version__ = "3.0.0"
__author__ = "AlphaGenesis Team"

# 모든 공개 클래스와 함수
__all__ = [
    # 설정
    'config',
    'UnifiedConfig',
    
    # 데이터
    'DataManager',
    
    # 백테스트
    'RealBacktestEngine',
    'BacktestResult',
    
    # 분석
    'StrategyAnalyzer',
    'PortfolioOptimizer',
    
    # 위험 관리
    'RiskManager',
    'DynamicLeverageManager',
    
    # ML
    'MLOptimizer',
    
    # 거래
    'PositionManager',
    'LiveTradingEngine',
]

# 모듈 정보
def get_module_info():
    """모듈 정보 반환"""
    return {
        'name': 'AlphaGenesis Core',
        'version': __version__,
        'author': __author__,
        'modules': __all__
    }

# 의존성 검사
def check_dependencies():
    """필수 의존성 검사"""
    missing_deps = []
    required_packages = [
        'pandas', 'numpy', 'ccxt', 'scikit-learn', 
        'xgboost', 'flask', 'requests'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_deps.append(package)
    
    return missing_deps

# 초기화 함수
def initialize_core(config_override=None):
    """코어 모듈 초기화"""
    try:
        if config_override:
            config.update_config('basic', config_override)
        
        # 의존성 검사
        missing = check_dependencies()
        if missing:
            print(f"경고: 일부 의존성 누락 - {missing}")
            print("pip install -r requirements.txt 실행을 권장합니다.")
        
        return True
        
    except Exception as e:
        print(f"코어 모듈 초기화 실패: {e}")
        return False

# 모듈 호환성 검사
def check_module_compatibility():
    """모듈간 호환성 검사"""
    compatibility_issues = []
    
    try:
        # 기본 설정 검증
        errors = config.validate_config()
        if errors:
            compatibility_issues.extend(errors)
            
    except Exception as e:
        compatibility_issues.append(f"설정 검증 실패: {e}")
    
    return compatibility_issues