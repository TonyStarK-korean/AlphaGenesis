import requests
import json
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import time
from enum import Enum

class NotificationType(Enum):
    """알림 타입"""
    TRADE_SIGNAL = "TRADE_SIGNAL"
    POSITION_UPDATE = "POSITION_UPDATE"
    EXCHANGE_RISK = "EXCHANGE_RISK"
    PERFORMANCE_REPORT = "PERFORMANCE_REPORT"
    SYSTEM_ALERT = "SYSTEM_ALERT"
    EMERGENCY = "EMERGENCY"

class TelegramNotificationBot:
    """
    텔레그램 알림 봇
    - 거래 신호 알림
    - 포지션 업데이트
    - 거래소 위험 경고
    - 성과 리포트
    - 시스템 알림
    """
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # 알림 설정
        self.notification_settings = {
            NotificationType.TRADE_SIGNAL: True,
            NotificationType.POSITION_UPDATE: True,
            NotificationType.EXCHANGE_RISK: True,
            NotificationType.PERFORMANCE_REPORT: True,
            NotificationType.SYSTEM_ALERT: True,
            NotificationType.EMERGENCY: True
        }
        
        # 알림 제한 (스팸 방지)
        self.rate_limits = {
            NotificationType.TRADE_SIGNAL: {'max_per_hour': 10, 'last_sent': {}},
            NotificationType.POSITION_UPDATE: {'max_per_hour': 5, 'last_sent': {}},
            NotificationType.EXCHANGE_RISK: {'max_per_hour': 3, 'last_sent': {}},
            NotificationType.PERFORMANCE_REPORT: {'max_per_hour': 1, 'last_sent': {}},
            NotificationType.SYSTEM_ALERT: {'max_per_hour': 5, 'last_sent': {}},
            NotificationType.EMERGENCY: {'max_per_hour': 10, 'last_sent': {}}
        }
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """메시지 전송"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                self.logger.error(f"텔레그램 전송 실패: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"텔레그램 전송 중 오류: {str(e)}")
            return False
            
    def send_trade_signal(self, signal_data: Dict) -> bool:
        """거래 신호 알림"""
        if not self._check_rate_limit(NotificationType.TRADE_SIGNAL, signal_data.get('symbol', 'general')):
            return False
            
        symbol = signal_data.get('symbol', 'Unknown')
        signal_type = signal_data.get('signal', 0)
        confidence = signal_data.get('confidence', 0.0)
        strength = signal_data.get('strength', 0.0)
        price = signal_data.get('price', 0.0)
        reason = signal_data.get('reason', 'Unknown')
        
        # 신호 타입에 따른 이모지
        if signal_type == 1:
            emoji = "🟢"
            action = "매수"
        elif signal_type == -1:
            emoji = "🔴"
            action = "매도"
        else:
            emoji = "⚪"
            action = "청산"
            
        message = f"""
{emoji} <b>거래 신호 감지</b> {emoji}

📊 <b>종목:</b> {symbol}
🎯 <b>행동:</b> {action}
📈 <b>현재가:</b> ${price:,.2f}
🎯 <b>신뢰도:</b> {confidence:.1%}
💪 <b>강도:</b> {strength:.1%}
📝 <b>사유:</b> {reason}

⏰ <b>시간:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return self.send_message(message)
        
    def send_position_update(self, position_data: Dict) -> bool:
        """포지션 업데이트 알림"""
        if not self._check_rate_limit(NotificationType.POSITION_UPDATE, position_data.get('symbol', 'general')):
            return False
            
        symbol = position_data.get('symbol', 'Unknown')
        action = position_data.get('action', 'Unknown')
        quantity = position_data.get('quantity', 0.0)
        price = position_data.get('price', 0.0)
        pnl = position_data.get('pnl', 0.0)
        pnl_percent = position_data.get('pnl_percent', 0.0)
        
        # 액션에 따른 이모지
        if action == 'OPEN':
            emoji = "📈"
        elif action == 'CLOSE':
            emoji = "📉"
        elif action == 'UPDATE':
            emoji = "📊"
        else:
            emoji = "📋"
            
        # PnL에 따른 색상
        if pnl > 0:
            pnl_emoji = "🟢"
        elif pnl < 0:
            pnl_emoji = "🔴"
        else:
            pnl_emoji = "⚪"
            
        message = f"""
{emoji} <b>포지션 업데이트</b> {emoji}

📊 <b>종목:</b> {symbol}
🎯 <b>액션:</b> {action}
📦 <b>수량:</b> {quantity:.4f}
💰 <b>가격:</b> ${price:,.2f}
{pnl_emoji} <b>손익:</b> ${pnl:,.2f} ({pnl_percent:+.2f}%)

⏰ <b>시간:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return self.send_message(message)
        
    def send_exchange_risk_alert(self, risk_data: Dict) -> bool:
        """거래소 위험 경고"""
        if not self._check_rate_limit(NotificationType.EXCHANGE_RISK, risk_data.get('exchange', 'general')):
            return False
            
        exchange = risk_data.get('exchange', 'Unknown')
        risk_level = risk_data.get('risk_level', 'Unknown')
        risk_score = risk_data.get('risk_score', 0.0)
        recommendation = risk_data.get('recommendation', 'Unknown')
        
        # 위험 수준에 따른 이모지
        if risk_level == 'CRITICAL':
            emoji = "🚨🚨🚨"
        elif risk_level == 'HIGH':
            emoji = "🚨🚨"
        elif risk_level == 'MEDIUM':
            emoji = "⚠️"
        else:
            emoji = "ℹ️"
            
        message = f"""
{emoji} <b>거래소 위험 경고</b> {emoji}

🏦 <b>거래소:</b> {exchange.upper()}
⚠️ <b>위험 수준:</b> {risk_level}
📊 <b>위험 점수:</b> {risk_score:.1%}
💡 <b>권고사항:</b> {recommendation}

🔍 <b>상세 정보:</b>
• 거래량 변화: {risk_data.get('volume_analysis', {}).get('volume_change_24h', 0):.2%}
• 출금 지연: {risk_data.get('withdrawal_analysis', {}).get('withdrawal_delay_hours', 0):.1f}시간
• 유동성 비율: {risk_data.get('liquidity_analysis', {}).get('liquidity_ratio', 0):.2%}
• 사용자 불만: {risk_data.get('complaint_analysis', {}).get('complaints_24h', 0)}건

⏰ <b>감지 시간:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ <b>즉시 대응 조치를 권고합니다!</b>
        """
        
        return self.send_message(message)
        
    def send_performance_report(self, performance_data: Dict) -> bool:
        """성과 리포트"""
        if not self._check_rate_limit(NotificationType.PERFORMANCE_REPORT, 'daily'):
            return False
            
        total_trades = performance_data.get('total_trades', 0)
        win_rate = performance_data.get('win_rate', 0.0)
        total_pnl = performance_data.get('total_pnl', 0.0)
        total_return = performance_data.get('total_return', 0.0)
        current_capital = performance_data.get('current_capital', 0.0)
        open_positions = performance_data.get('open_positions', 0)
        
        # 성과에 따른 이모지
        if total_return > 5:
            emoji = "🚀"
        elif total_return > 0:
            emoji = "📈"
        elif total_return > -5:
            emoji = "📊"
        else:
            emoji = "📉"
            
        message = f"""
{emoji} <b>일일 성과 리포트</b> {emoji}

📊 <b>거래 통계:</b>
• 총 거래: {total_trades}건
• 승률: {win_rate:.1f}%
• 총 손익: ${total_pnl:,.2f}
• 총 수익률: {total_return:+.2f}%

💰 <b>자본 현황:</b>
• 현재 자본: ${current_capital:,.2f}
• 오픈 포지션: {open_positions}개

📅 <b>보고 기간:</b> {datetime.now().strftime('%Y-%m-%d')}

{emoji} <b>시스템이 정상적으로 운영 중입니다!</b>
        """
        
        return self.send_message(message)
        
    def send_system_alert(self, alert_data: Dict) -> bool:
        """시스템 알림"""
        if not self._check_rate_limit(NotificationType.SYSTEM_ALERT, alert_data.get('type', 'general')):
            return False
            
        alert_type = alert_data.get('type', 'Unknown')
        message_text = alert_data.get('message', 'Unknown')
        severity = alert_data.get('severity', 'INFO')
        
        # 심각도에 따른 이모지
        if severity == 'CRITICAL':
            emoji = "🚨"
        elif severity == 'ERROR':
            emoji = "❌"
        elif severity == 'WARNING':
            emoji = "⚠️"
        else:
            emoji = "ℹ️"
            
        message = f"""
{emoji} <b>시스템 알림</b> {emoji}

🔧 <b>타입:</b> {alert_type}
📝 <b>메시지:</b> {message_text}
⚠️ <b>심각도:</b> {severity}

⏰ <b>시간:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return self.send_message(message)
        
    def send_emergency_alert(self, emergency_data: Dict) -> bool:
        """긴급 알림"""
        if not self._check_rate_limit(NotificationType.EMERGENCY, emergency_data.get('type', 'general')):
            return False
            
        emergency_type = emergency_data.get('type', 'Unknown')
        message_text = emergency_data.get('message', 'Unknown')
        action_required = emergency_data.get('action_required', 'Unknown')
        
        message = f"""
🚨🚨🚨 <b>긴급 알림</b> 🚨🚨🚨

🚨 <b>긴급 상황:</b> {emergency_type}
📝 <b>상세:</b> {message_text}
⚡ <b>필요 조치:</b> {action_required}

⏰ <b>시간:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🚨 <b>즉시 대응이 필요합니다!</b> 🚨
        """
        
        return self.send_message(message)
        
    def _check_rate_limit(self, notification_type: NotificationType, key: str) -> bool:
        """알림 제한 확인"""
        
        if not self.notification_settings.get(notification_type, True):
            return False
            
        rate_limit = self.rate_limits.get(notification_type, {})
        max_per_hour = rate_limit.get('max_per_hour', 10)
        last_sent = rate_limit.get('last_sent', {})
        
        current_time = datetime.now()
        one_hour_ago = current_time - timedelta(hours=1)
        
        # 해당 키의 마지막 전송 시간 확인
        if key in last_sent:
            if last_sent[key] > one_hour_ago:
                return False
                
        # 시간당 전송 횟수 확인
        recent_sends = [time for time in last_sent.values() if time > one_hour_ago]
        if len(recent_sends) >= max_per_hour:
            return False
            
        # 전송 시간 기록
        last_sent[key] = current_time
        
        return True
        
    def send_custom_message(self, message: str, notification_type: NotificationType = NotificationType.SYSTEM_ALERT) -> bool:
        """사용자 정의 메시지 전송"""
        return self.send_message(message)
        
    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                bot_info = response.json()
                self.logger.info(f"봇 연결 성공: {bot_info['result']['username']}")
                return True
            else:
                self.logger.error(f"봇 연결 실패: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"봇 연결 테스트 중 오류: {str(e)}")
            return False
            
    def get_chat_info(self) -> Optional[Dict]:
        """채팅 정보 조회"""
        try:
            url = f"{self.base_url}/getChat"
            data = {'chat_id': self.chat_id}
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                return response.json()['result']
            else:
                self.logger.error(f"채팅 정보 조회 실패: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"채팅 정보 조회 중 오류: {str(e)}")
            return None 