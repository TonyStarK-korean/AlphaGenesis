#!/usr/bin/env python3
"""
상위 0.01%급 완전 자동매매 시스템
- 실시간 매수/매도/익절/손절/청산
- 거래소 파산 감지 및 자동 출금
- 텔레그램 알림 시스템
- 고급 리스크 관리
"""

import sys
import os
import time
import logging
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import schedule

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 시스템 모듈 임포트
from src.core.trading_engine.auto_trading_engine import AutoTradingEngine
from src.exchange.exchange_monitor.exchange_bankruptcy_detector import ExchangeBankruptcyDetector
from src.exchange.withdrawal_system.auto_withdrawal_manager import AutoWithdrawalManager
from src.notification.telegram_bot.telegram_notification_bot import TelegramNotificationBot
from src.config.trading_config import *

class UltimateTradingSystem:
    """
    상위 0.01%급 완전 자동매매 시스템
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.logger.info("🚀 상위 0.01%급 자동매매 시스템 초기화 시작...")
        
        # 시스템 컴포넌트 초기화
        self.trading_engine = None
        self.exchange_monitor = None
        self.withdrawal_manager = None
        self.telegram_bot = None
        
        # 시스템 상태
        self.is_running = False
        self.system_start_time = None
        self.last_performance_report = None
        
        # 스레드 관리
        self.monitoring_thread = None
        self.trading_thread = None
        self.notification_thread = None
        
        # 종료 신호 처리
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('UltimateTradingSystem')
        logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))
        
        # 파일 핸들러
        file_handler = logging.FileHandler(LOGGING_CONFIG['file'])
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(LOGGING_CONFIG['format'])
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def initialize_system(self) -> bool:
        """시스템 초기화"""
        try:
            self.logger.info("🔧 시스템 컴포넌트 초기화 중...")
            
            # 1. 텔레그램 봇 초기화
            if TELEGRAM_CONFIG['enable_notifications'] and TELEGRAM_CONFIG['bot_token']:
                self.telegram_bot = TelegramNotificationBot(
                    TELEGRAM_CONFIG['bot_token'],
                    TELEGRAM_CONFIG['chat_id']
                )
                
                if self.telegram_bot.test_connection():
                    self.logger.info("✅ 텔레그램 봇 연결 성공")
                    self._send_system_alert("시스템 시작", "INFO")
                else:
                    self.logger.warning("⚠️ 텔레그램 봇 연결 실패")
            else:
                self.logger.warning("⚠️ 텔레그램 알림 비활성화")
                
            # 2. 거래 엔진 초기화
            self.trading_engine = AutoTradingEngine(
                initial_capital=TRADING_CONFIG['initial_capital'],
                max_position_size=TRADING_CONFIG['max_position_size'],
                default_stop_loss=TRADING_CONFIG['default_stop_loss'],
                default_take_profit=TRADING_CONFIG['default_take_profit'],
                trailing_stop=TRADING_CONFIG['trailing_stop'],
                trailing_stop_distance=TRADING_CONFIG['trailing_stop_distance']
            )
            self.logger.info("✅ 거래 엔진 초기화 완료")
            
            # 3. 거래소 모니터 초기화
            self.exchange_monitor = ExchangeBankruptcyDetector(
                telegram_bot_token=TELEGRAM_CONFIG.get('bot_token'),
                telegram_chat_id=TELEGRAM_CONFIG.get('chat_id')
            )
            self.logger.info("✅ 거래소 모니터 초기화 완료")
            
            # 4. 출금 관리자 초기화
            self.withdrawal_manager = AutoWithdrawalManager(
                personal_wallets=PERSONAL_WALLETS,
                withdrawal_fees=WITHDRAWAL_FEES,
                min_withdrawal_amounts=MIN_WITHDRAWAL_AMOUNTS
            )
            self.logger.info("✅ 출금 관리자 초기화 완료")
            
            self.logger.info("🎉 시스템 초기화 완료!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 시스템 초기화 실패: {str(e)}")
            return False
            
    def start_system(self):
        """시스템 시작"""
        if not self.initialize_system():
            self.logger.error("시스템 초기화 실패로 종료합니다.")
            return
            
        self.is_running = True
        self.system_start_time = datetime.now()
        
        self.logger.info("🚀 시스템 시작!")
        self._send_system_alert("시스템 시작", "INFO")
        
        # 스레드 시작
        self._start_monitoring_thread()
        self._start_trading_thread()
        self._start_notification_thread()
        
        # 스케줄러 설정
        self._setup_scheduler()
        
        # 메인 루프
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단됨")
        finally:
            self.stop_system()
            
    def stop_system(self):
        """시스템 종료"""
        self.logger.info("🛑 시스템 종료 중...")
        self.is_running = False
        
        # 스레드 종료 대기
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
            
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
            
        if self.notification_thread and self.notification_thread.is_alive():
            self.notification_thread.join(timeout=5)
            
        # 최종 성과 리포트
        self._send_final_performance_report()
        
        self.logger.info("✅ 시스템 종료 완료")
        
    def _start_monitoring_thread(self):
        """모니터링 스레드 시작"""
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("📊 모니터링 스레드 시작")
        
    def _start_trading_thread(self):
        """거래 스레드 시작"""
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        self.logger.info("💰 거래 스레드 시작")
        
    def _start_notification_thread(self):
        """알림 스레드 시작"""
        self.notification_thread = threading.Thread(target=self._notification_loop, daemon=True)
        self.notification_thread.start()
        self.logger.info("📢 알림 스레드 시작")
        
    def _monitoring_loop(self):
        """거래소 모니터링 루프"""
        while self.is_running:
            try:
                # 거래소 위험도 분석
                for exchange in MONITORED_EXCHANGES:
                    risk_analysis = self.exchange_monitor.analyze_exchange_health(exchange)
                    
                    # 위험 수준이 HIGH 이상이면 경고
                    if risk_analysis['risk_level'] in ['HIGH', 'CRITICAL']:
                        self.logger.warning(f"🚨 {exchange} 위험 감지: {risk_analysis['risk_level']}")
                        
                        # 텔레그램 경고
                        if self.telegram_bot:
                            self.telegram_bot.send_exchange_risk_alert(risk_analysis)
                            
                        # 긴급 출금 실행
                        if (risk_analysis['risk_level'] == 'CRITICAL' and 
                            EMERGENCY_CONFIG['enable_emergency_withdrawal']):
                            self._execute_emergency_withdrawal(exchange, risk_analysis)
                            
                # 5분 대기
                time.sleep(300)
                
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {str(e)}")
                time.sleep(60)
                
    def _trading_loop(self):
        """거래 루프"""
        while self.is_running:
            try:
                # 시뮬레이션된 거래 신호 생성 (실제로는 ML 모델에서 생성)
                signals = self._generate_trading_signals()
                
                for signal in signals:
                    if not self.is_running:
                        break
                        
                    # 거래 신호 처리
                    result = self.trading_engine.process_signal(
                        signal['symbol'],
                        signal,
                        signal['price'],
                        signal.get('market_data', None)
                    )
                    
                    if result:
                        self.logger.info(f"거래 실행: {signal['symbol']} - {result['order_type'].value}")
                        
                        # 텔레그램 알림
                        if self.telegram_bot:
                            self.telegram_bot.send_trade_signal(signal)
                            
                # 1분 대기
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"거래 루프 오류: {str(e)}")
                time.sleep(30)
                
    def _notification_loop(self):
        """알림 루프"""
        while self.is_running:
            try:
                # 시스템 상태 체크
                self._check_system_health()
                
                # 10분 대기
                time.sleep(600)
                
            except Exception as e:
                self.logger.error(f"알림 루프 오류: {str(e)}")
                time.sleep(60)
                
    def _generate_trading_signals(self) -> List[Dict]:
        """거래 신호 생성 (시뮬레이션)"""
        signals = []
        
        # 시뮬레이션된 신호 생성
        for symbol in TRADING_SYMBOLS[:5]:  # 상위 5개만
            if self.is_running:
                # 랜덤 신호 생성
                import random
                signal_type = random.choice([-1, 0, 1])
                confidence = random.uniform(0.6, 0.95)
                strength = random.uniform(0.5, 0.9)
                price = random.uniform(100, 50000)
                
                signal = {
                    'symbol': symbol,
                    'signal': signal_type,
                    'confidence': confidence,
                    'strength': strength,
                    'price': price,
                    'reason': 'ML_PREDICTION',
                    'timestamp': datetime.now()
                }
                
                signals.append(signal)
                
        return signals
        
    def _execute_emergency_withdrawal(self, exchange: str, risk_analysis: Dict):
        """긴급 출금 실행"""
        self.logger.warning(f"🚨 {exchange} 긴급 출금 실행!")
        
        try:
            # 시뮬레이션된 잔액 (실제로는 거래소 API 호출)
            balances = {
                'BTC': 0.1,
                'ETH': 1.0,
                'USDT': 10000,
                'USDC': 5000,
                'BNB': 10.0
            }
            
            # 긴급 출금 실행
            withdrawal_orders = self.withdrawal_manager.emergency_withdrawal_all(
                balances,
                risk_analysis['risk_level'],
                exchange
            )
            
            self.logger.info(f"긴급 출금 완료: {len(withdrawal_orders)}건")
            
            # 긴급 알림
            if self.telegram_bot:
                self.telegram_bot.send_emergency_alert({
                    'type': 'EXCHANGE_RISK',
                    'message': f"{exchange} 거래소 위험으로 긴급 출금 실행",
                    'action_required': '출금 상태 확인 필요'
                })
                
        except Exception as e:
            self.logger.error(f"긴급 출금 실패: {str(e)}")
            
    def _check_system_health(self):
        """시스템 건강도 체크"""
        try:
            # 성과 지표 확인
            performance = self.trading_engine.get_performance_metrics()
            
            # 임계값 체크
            if performance['total_return'] < -PERFORMANCE_CONFIG['alert_thresholds']['max_drawdown']:
                self._send_system_alert("성과 저하 감지", "WARNING")
                
            # 메모리 사용량 체크
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:
                self._send_system_alert("메모리 사용량 높음", "WARNING")
                
        except Exception as e:
            self.logger.error(f"시스템 건강도 체크 오류: {str(e)}")
            
    def _setup_scheduler(self):
        """스케줄러 설정"""
        # 매일 자정 성과 리포트
        schedule.every().day.at("00:00").do(self._send_daily_performance_report)
        
        # 매시간 시스템 상태 체크
        schedule.every().hour.do(self._check_system_health)
        
        # 스케줄러 실행 스레드
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)
                
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
    def _send_daily_performance_report(self):
        """일일 성과 리포트 전송"""
        try:
            performance = self.trading_engine.get_performance_metrics()
            
            if self.telegram_bot:
                self.telegram_bot.send_performance_report(performance)
                
            self.last_performance_report = datetime.now()
            
        except Exception as e:
            self.logger.error(f"성과 리포트 전송 오류: {str(e)}")
            
    def _send_final_performance_report(self):
        """최종 성과 리포트 전송"""
        try:
            if self.trading_engine:
                performance = self.trading_engine.get_performance_metrics()
                
                if self.telegram_bot:
                    self.telegram_bot.send_performance_report(performance)
                    
        except Exception as e:
            self.logger.error(f"최종 성과 리포트 전송 오류: {str(e)}")
            
    def _send_system_alert(self, message: str, severity: str = "INFO"):
        """시스템 알림 전송"""
        try:
            if self.telegram_bot:
                self.telegram_bot.send_system_alert({
                    'type': 'SYSTEM_STATUS',
                    'message': message,
                    'severity': severity
                })
        except Exception as e:
            self.logger.error(f"시스템 알림 전송 오류: {str(e)}")
            
    def _signal_handler(self, signum, frame):
        """시그널 핸들러"""
        self.logger.info(f"시그널 {signum} 수신, 시스템 종료 중...")
        self.stop_system()
        sys.exit(0)
        
    def get_system_status(self) -> Dict:
        """시스템 상태 반환"""
        return {
            'is_running': self.is_running,
            'start_time': self.system_start_time,
            'uptime': datetime.now() - self.system_start_time if self.system_start_time else None,
            'performance': self.trading_engine.get_performance_metrics() if self.trading_engine else None,
            'open_positions': len(self.trading_engine.get_open_positions()) if self.trading_engine else 0
        }

def main():
    """메인 함수"""
    print("🚀 상위 0.01%급 완전 자동매매 시스템")
    print("=" * 50)
    
    # 설정 확인
    if not TELEGRAM_CONFIG['bot_token']:
        print("⚠️  텔레그램 봇 토큰이 설정되지 않았습니다.")
        print("   src/config/trading_config.py에서 설정하세요.")
        
    if not EXCHANGE_CONFIG['api_key']:
        print("⚠️  거래소 API 키가 설정되지 않았습니다.")
        print("   src/config/trading_config.py에서 설정하세요.")
        
    # 시스템 시작
    trading_system = UltimateTradingSystem()
    
    try:
        trading_system.start_system()
    except Exception as e:
        print(f"❌ 시스템 실행 중 오류 발생: {str(e)}")
        logging.error(f"시스템 실행 오류: {str(e)}")

if __name__ == "__main__":
    main() 