"""
AlphaGenesis 공통 예외 클래스
"""

class AlphaGenesisException(Exception):
    """AlphaGenesis 기본 예외 클래스"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

class DataError(AlphaGenesisException):
    """데이터 관련 예외"""
    pass

class DataNotFoundError(DataError):
    """데이터를 찾을 수 없음"""
    pass

class DataValidationError(DataError):
    """데이터 검증 실패"""
    pass

class DataCorruptionError(DataError):
    """데이터 손상"""
    pass

class TradingError(AlphaGenesisException):
    """거래 관련 예외"""
    pass

class InvalidSignalError(TradingError):
    """유효하지 않은 신호"""
    pass

class PositionSizingError(TradingError):
    """포지션 크기 설정 오류"""
    pass

class RiskManagementError(TradingError):
    """리스크 관리 오류"""
    pass

class ExchangeError(AlphaGenesisException):
    """거래소 관련 예외"""
    pass

class ExchangeConnectionError(ExchangeError):
    """거래소 연결 오류"""
    pass

class ExchangeAPIError(ExchangeError):
    """거래소 API 오류"""
    pass

class InsufficientFundsError(ExchangeError):
    """잔액 부족"""
    pass

class MLModelError(AlphaGenesisException):
    """머신러닝 모델 관련 예외"""
    pass

class ModelNotTrainedError(MLModelError):
    """모델이 훈련되지 않음"""
    pass

class ModelValidationError(MLModelError):
    """모델 검증 실패"""
    pass

class FeatureEngineeringError(MLModelError):
    """피처 엔지니어링 오류"""
    pass

class BacktestError(AlphaGenesisException):
    """백테스트 관련 예외"""
    pass

class BacktestDataError(BacktestError):
    """백테스트 데이터 오류"""
    pass

class BacktestConfigError(BacktestError):
    """백테스트 설정 오류"""
    pass

class StrategyError(AlphaGenesisException):
    """전략 관련 예외"""
    pass

class StrategyNotFoundError(StrategyError):
    """전략을 찾을 수 없음"""
    pass

class StrategyValidationError(StrategyError):
    """전략 검증 실패"""
    pass

class ConfigurationError(AlphaGenesisException):
    """설정 관련 예외"""
    pass

class InvalidConfigError(ConfigurationError):
    """유효하지 않은 설정"""
    pass

class MissingConfigError(ConfigurationError):
    """설정 누락"""
    pass

class SystemError(AlphaGenesisException):
    """시스템 관련 예외"""
    pass

class ResourceExhaustedError(SystemError):
    """리소스 부족"""
    pass

class TimeoutError(SystemError):
    """타임아웃"""
    pass

class NetworkError(SystemError):
    """네트워크 오류"""
    pass

def handle_exception(func):
    """예외 처리 데코레이터"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AlphaGenesisException:
            # AlphaGenesis 예외는 다시 발생
            raise
        except Exception as e:
            # 기타 예외는 AlphaGenesisException으로 래핑
            raise AlphaGenesisException(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={
                    'function': func.__name__,
                    'args': args,
                    'kwargs': kwargs,
                    'original_error': str(e)
                }
            )
    return wrapper

def log_exception(logger):
    """예외 로깅 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AlphaGenesisException as e:
                logger.error(f"AlphaGenesis Error in {func.__name__}: {e}")
                if e.details:
                    logger.error(f"Details: {e.details}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                raise AlphaGenesisException(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    error_code="UNEXPECTED_ERROR"
                )
        return wrapper
    return decorator