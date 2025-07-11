"""
통합 로깅 설정
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

class ColoredFormatter(logging.Formatter):
    """컬러 포맷터"""
    
    # 컬러 코드
    COLORS = {
        'DEBUG': '\033[36m',    # 청록색
        'INFO': '\033[32m',     # 녹색
        'WARNING': '\033[33m',  # 노란색
        'ERROR': '\033[31m',    # 빨간색
        'CRITICAL': '\033[35m', # 자홍색
        'RESET': '\033[0m'      # 리셋
    }
    
    def format(self, record):
        # 기본 포맷 적용
        formatted = super().format(record)
        
        # 컬러 적용
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        return f"{color}{formatted}{reset}"

class JsonFormatter(logging.Formatter):
    """JSON 포맷터"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # 예외 정보가 있으면 추가
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # 추가 필드가 있으면 포함
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        return json.dumps(log_entry, ensure_ascii=False)

class LoggerManager:
    """로거 관리자"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 로거 설정
        self.loggers = {}
        self._setup_root_logger()
        
    def _setup_root_logger(self):
        """루트 로거 설정"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
    def get_logger(self, name: str, level: str = "INFO") -> logging.Logger:
        """로거 반환"""
        if name in self.loggers:
            return self.loggers[name]
            
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # 핸들러 추가
        self._add_console_handler(logger)
        self._add_file_handler(logger, name)
        self._add_error_handler(logger, name)
        
        # 전파 방지
        logger.propagate = False
        
        self.loggers[name] = logger
        return logger
        
    def _add_console_handler(self, logger: logging.Logger):
        """콘솔 핸들러 추가"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 컬러 포맷터 사용
        formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
    def _add_file_handler(self, logger: logging.Logger, name: str):
        """파일 핸들러 추가"""
        file_path = self.log_dir / f"{name}.log"
        
        # 로테이팅 파일 핸들러
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # 일반 포맷터 사용
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
    def _add_error_handler(self, logger: logging.Logger, name: str):
        """에러 전용 핸들러 추가"""
        error_file_path = self.log_dir / f"{name}_errors.log"
        
        error_handler = logging.handlers.RotatingFileHandler(
            error_file_path,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        # JSON 포맷터 사용
        formatter = JsonFormatter()
        error_handler.setFormatter(formatter)
        
        logger.addHandler(error_handler)
        
    def setup_trading_logger(self) -> logging.Logger:
        """거래 전용 로거 설정"""
        logger = self.get_logger("trading", "DEBUG")
        
        # 거래 로그 파일 핸들러 추가
        trade_file_path = self.log_dir / "trades.log"
        trade_handler = logging.handlers.TimedRotatingFileHandler(
            trade_file_path,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        trade_handler.setLevel(logging.INFO)
        
        # 거래 로그 포맷터
        trade_formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        trade_handler.setFormatter(trade_formatter)
        
        logger.addHandler(trade_handler)
        return logger
        
    def setup_backtest_logger(self) -> logging.Logger:
        """백테스트 전용 로거 설정"""
        logger = self.get_logger("backtest", "DEBUG")
        
        # 백테스트 로그 파일 핸들러 추가
        backtest_file_path = self.log_dir / "backtest.log"
        backtest_handler = logging.handlers.RotatingFileHandler(
            backtest_file_path,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        backtest_handler.setLevel(logging.DEBUG)
        
        # 상세 포맷터
        backtest_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        backtest_handler.setFormatter(backtest_formatter)
        
        logger.addHandler(backtest_handler)
        return logger
        
    def setup_ml_logger(self) -> logging.Logger:
        """ML 전용 로거 설정"""
        logger = self.get_logger("ml", "DEBUG")
        
        # ML 로그 파일 핸들러 추가
        ml_file_path = self.log_dir / "ml.log"
        ml_handler = logging.handlers.RotatingFileHandler(
            ml_file_path,
            maxBytes=20*1024*1024,  # 20MB
            backupCount=5,
            encoding='utf-8'
        )
        ml_handler.setLevel(logging.DEBUG)
        
        # ML 포맷터
        ml_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ml_handler.setFormatter(ml_formatter)
        
        logger.addHandler(ml_handler)
        return logger
        
    def setup_system_logger(self) -> logging.Logger:
        """시스템 전용 로거 설정"""
        logger = self.get_logger("system", "INFO")
        
        # 시스템 로그 파일 핸들러 추가
        system_file_path = self.log_dir / "system.log"
        system_handler = logging.handlers.TimedRotatingFileHandler(
            system_file_path,
            when='midnight',
            interval=1,
            backupCount=7,
            encoding='utf-8'
        )
        system_handler.setLevel(logging.INFO)
        
        # 시스템 포맷터
        system_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        system_handler.setFormatter(system_formatter)
        
        logger.addHandler(system_handler)
        return logger
        
    def log_trade(self, logger: logging.Logger, trade_info: dict):
        """거래 로그 기록"""
        logger.info(f"TRADE: {json.dumps(trade_info, ensure_ascii=False)}")
        
    def log_signal(self, logger: logging.Logger, signal_info: dict):
        """신호 로그 기록"""
        logger.info(f"SIGNAL: {json.dumps(signal_info, ensure_ascii=False)}")
        
    def log_performance(self, logger: logging.Logger, performance_info: dict):
        """성능 로그 기록"""
        logger.info(f"PERFORMANCE: {json.dumps(performance_info, ensure_ascii=False)}")
        
    def log_error_with_context(self, logger: logging.Logger, error: Exception, context: dict):
        """컨텍스트와 함께 에러 로그 기록"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        logger.error(f"ERROR: {json.dumps(error_info, ensure_ascii=False)}")

# 전역 로거 매니저
logger_manager = LoggerManager()

# 자주 사용하는 로거들
main_logger = logger_manager.get_logger("main")
trading_logger = logger_manager.setup_trading_logger()
backtest_logger = logger_manager.setup_backtest_logger()
ml_logger = logger_manager.setup_ml_logger()
system_logger = logger_manager.setup_system_logger()

# 편의 함수들
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """로거 반환"""
    return logger_manager.get_logger(name, level)

def log_trade(trade_info: dict):
    """거래 로그 기록"""
    logger_manager.log_trade(trading_logger, trade_info)

def log_signal(signal_info: dict):
    """신호 로그 기록"""
    logger_manager.log_signal(trading_logger, signal_info)

def log_performance(performance_info: dict):
    """성능 로그 기록"""
    logger_manager.log_performance(trading_logger, performance_info)

def log_error_with_context(error: Exception, context: dict):
    """컨텍스트와 함께 에러 로그 기록"""
    logger_manager.log_error_with_context(main_logger, error, context)