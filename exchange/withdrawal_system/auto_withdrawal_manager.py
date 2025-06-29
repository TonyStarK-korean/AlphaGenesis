import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
import time
from enum import Enum

class WithdrawalStatus(Enum):
    """출금 상태"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class AutoWithdrawalManager:
    """
    자동 출금 관리 시스템
    - 거래소 파산 위험 감지 시 자동 출금
    - 개인 지갑으로 안전한 이체
    - 출금 상태 모니터링
    - 수수료 최적화
    """
    
    def __init__(self, 
                 personal_wallets: Dict[str, str],
                 withdrawal_fees: Dict[str, float] = None,
                 min_withdrawal_amounts: Dict[str, float] = None):
        
        # 개인 지갑 주소
        self.personal_wallets = personal_wallets
        
        # 출금 수수료 (기본값)
        self.withdrawal_fees = withdrawal_fees or {
            'BTC': 0.0005,
            'ETH': 0.005,
            'USDT': 1.0,
            'USDC': 1.0,
            'BNB': 0.001,
            'ADA': 1.0,
            'DOT': 0.1,
            'LINK': 0.1,
            'LTC': 0.001,
            'BCH': 0.001
        }
        
        # 최소 출금 금액
        self.min_withdrawal_amounts = min_withdrawal_amounts or {
            'BTC': 0.001,
            'ETH': 0.01,
            'USDT': 10.0,
            'USDC': 10.0,
            'BNB': 0.01,
            'ADA': 10.0,
            'DOT': 1.0,
            'LINK': 1.0,
            'LTC': 0.01,
            'BCH': 0.01
        }
        
        # 출금 기록
        self.withdrawal_history = []
        self.pending_withdrawals = {}
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def emergency_withdrawal_all(self, 
                                exchange_balances: Dict[str, float],
                                risk_level: str,
                                exchange_name: str) -> List[Dict]:
        """긴급 전체 출금"""
        
        self.logger.warning(f"긴급 전체 출금 시작: {exchange_name} - {risk_level}")
        
        withdrawal_orders = []
        
        for symbol, balance in exchange_balances.items():
            if balance <= 0:
                continue
                
            # 최소 출금 금액 확인
            min_amount = self.min_withdrawal_amounts.get(symbol, 0)
            if balance < min_amount:
                self.logger.warning(f"{symbol} 잔액 부족: {balance} < {min_amount}")
                continue
                
            # 개인 지갑 주소 확인
            wallet_address = self.personal_wallets.get(symbol)
            if not wallet_address:
                self.logger.warning(f"{symbol} 개인 지갑 주소 없음")
                continue
                
            # 출금 주문 생성
            withdrawal_order = self._create_withdrawal_order(
                symbol=symbol,
                amount=balance,
                wallet_address=wallet_address,
                exchange_name=exchange_name,
                reason=f"EMERGENCY_{risk_level}"
            )
            
            withdrawal_orders.append(withdrawal_order)
            
        # 출금 우선순위 설정
        withdrawal_orders = self._prioritize_withdrawals(withdrawal_orders)
        
        # 출금 실행
        for order in withdrawal_orders:
            try:
                result = self._execute_withdrawal(order)
                if result['status'] == WithdrawalStatus.COMPLETED:
                    self.logger.info(f"출금 완료: {order['symbol']} {order['amount']}")
                else:
                    self.logger.error(f"출금 실패: {order['symbol']} - {result['error']}")
                    
            except Exception as e:
                self.logger.error(f"출금 실행 중 오류: {order['symbol']} - {str(e)}")
                
        return withdrawal_orders
        
    def _create_withdrawal_order(self,
                                symbol: str,
                                amount: float,
                                wallet_address: str,
                                exchange_name: str,
                                reason: str) -> Dict:
        """출금 주문 생성"""
        
        # 수수료 계산
        fee = self.withdrawal_fees.get(symbol, 0)
        net_amount = amount - fee
        
        if net_amount <= 0:
            raise ValueError(f"수수료가 잔액보다 큽니다: {symbol}")
            
        order = {
            'id': f"WD_{int(time.time())}_{symbol}",
            'symbol': symbol,
            'amount': amount,
            'fee': fee,
            'net_amount': net_amount,
            'wallet_address': wallet_address,
            'exchange_name': exchange_name,
            'reason': reason,
            'status': WithdrawalStatus.PENDING,
            'created_at': datetime.now(),
            'estimated_completion': datetime.now() + timedelta(hours=1),
            'retry_count': 0,
            'max_retries': 3
        }
        
        return order
        
    def _prioritize_withdrawals(self, withdrawal_orders: List[Dict]) -> List[Dict]:
        """출금 우선순위 설정"""
        
        # 우선순위 점수 계산
        for order in withdrawal_orders:
            priority_score = 0
            
            # 1. 시장 가치 기준 (높은 가치 코인 우선)
            market_values = {
                'BTC': 100, 'ETH': 90, 'USDT': 80, 'USDC': 80,
                'BNB': 70, 'ADA': 60, 'DOT': 60, 'LINK': 50,
                'LTC': 40, 'BCH': 40
            }
            priority_score += market_values.get(order['symbol'], 10)
            
            # 2. 유동성 기준 (유동성이 낮은 코인 우선)
            liquidity_scores = {
                'USDT': 10, 'USDC': 10, 'BTC': 20, 'ETH': 20,
                'BNB': 30, 'ADA': 40, 'DOT': 40, 'LINK': 50,
                'LTC': 60, 'BCH': 60
            }
            priority_score += liquidity_scores.get(order['symbol'], 50)
            
            # 3. 금액 기준 (큰 금액 우선)
            amount_score = min(order['amount'] * 100, 50)
            priority_score += amount_score
            
            order['priority_score'] = priority_score
            
        # 우선순위 순으로 정렬
        withdrawal_orders.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return withdrawal_orders
        
    def _execute_withdrawal(self, order: Dict) -> Dict:
        """출금 실행 (시뮬레이션)"""
        
        try:
            # 실제 구현에서는 거래소 API 호출
            # 여기서는 시뮬레이션
            
            # 출금 처리 시간 (시뮬레이션)
            processing_time = np.random.exponential(0.5)  # 평균 30분
            
            # 성공 확률 (시뮬레이션)
            success_probability = 0.95  # 95% 성공률
            
            if np.random.random() < success_probability:
                # 성공
                order['status'] = WithdrawalStatus.COMPLETED
                order['completed_at'] = datetime.now()
                order['tx_hash'] = f"0x{np.random.bytes(32).hex()}"
                
                # 출금 기록에 추가
                self.withdrawal_history.append(order.copy())
                
                return {
                    'status': WithdrawalStatus.COMPLETED,
                    'tx_hash': order['tx_hash'],
                    'completed_at': order['completed_at']
                }
            else:
                # 실패
                order['status'] = WithdrawalStatus.FAILED
                order['error'] = "출금 처리 실패"
                order['retry_count'] += 1
                
                return {
                    'status': WithdrawalStatus.FAILED,
                    'error': order['error']
                }
                
        except Exception as e:
            order['status'] = WithdrawalStatus.FAILED
            order['error'] = str(e)
            order['retry_count'] += 1
            
            return {
                'status': WithdrawalStatus.FAILED,
                'error': str(e)
            }
            
    def retry_failed_withdrawals(self) -> List[Dict]:
        """실패한 출금 재시도"""
        
        retry_orders = []
        
        for order in self.withdrawal_history:
            if (order['status'] == WithdrawalStatus.FAILED and 
                order['retry_count'] < order['max_retries']):
                
                self.logger.info(f"출금 재시도: {order['symbol']} {order['amount']}")
                
                # 재시도 주문 생성
                retry_order = order.copy()
                retry_order['id'] = f"WD_RETRY_{int(time.time())}_{order['symbol']}"
                retry_order['created_at'] = datetime.now()
                retry_order['estimated_completion'] = datetime.now() + timedelta(hours=1)
                
                # 재시도 실행
                result = self._execute_withdrawal(retry_order)
                
                if result['status'] == WithdrawalStatus.COMPLETED:
                    self.logger.info(f"재시도 성공: {order['symbol']}")
                else:
                    self.logger.error(f"재시도 실패: {order['symbol']} - {result['error']}")
                    
                retry_orders.append(retry_order)
                
        return retry_orders
        
    def get_withdrawal_status(self, withdrawal_id: str) -> Optional[Dict]:
        """출금 상태 조회"""
        
        for order in self.withdrawal_history:
            if order['id'] == withdrawal_id:
                return order
                
        return None
        
    def get_withdrawal_summary(self) -> Dict:
        """출금 요약 정보"""
        
        total_withdrawals = len(self.withdrawal_history)
        completed_withdrawals = len([w for w in self.withdrawal_history if w['status'] == WithdrawalStatus.COMPLETED])
        failed_withdrawals = len([w for w in self.withdrawal_history if w['status'] == WithdrawalStatus.FAILED])
        
        total_amount = sum(w['amount'] for w in self.withdrawal_history if w['status'] == WithdrawalStatus.COMPLETED)
        total_fees = sum(w['fee'] for w in self.withdrawal_history if w['status'] == WithdrawalStatus.COMPLETED)
        
        # 코인별 출금 통계
        coin_stats = {}
        for withdrawal in self.withdrawal_history:
            symbol = withdrawal['symbol']
            if symbol not in coin_stats:
                coin_stats[symbol] = {
                    'total_amount': 0,
                    'total_fees': 0,
                    'count': 0
                }
                
            if withdrawal['status'] == WithdrawalStatus.COMPLETED:
                coin_stats[symbol]['total_amount'] += withdrawal['amount']
                coin_stats[symbol]['total_fees'] += withdrawal['fee']
                coin_stats[symbol]['count'] += 1
                
        return {
            'total_withdrawals': total_withdrawals,
            'completed_withdrawals': completed_withdrawals,
            'failed_withdrawals': failed_withdrawals,
            'success_rate': (completed_withdrawals / total_withdrawals * 100) if total_withdrawals > 0 else 0,
            'total_amount': total_amount,
            'total_fees': total_fees,
            'net_amount': total_amount - total_fees,
            'coin_stats': coin_stats
        }
        
    def estimate_withdrawal_time(self, symbol: str, amount: float) -> Dict:
        """출금 예상 시간 계산"""
        
        # 기본 처리 시간
        base_processing_time = 30  # 분
        
        # 코인별 추가 시간
        additional_times = {
            'BTC': 60,   # 1시간
            'ETH': 45,   # 45분
            'USDT': 15,  # 15분
            'USDC': 15,  # 15분
            'BNB': 30,   # 30분
            'ADA': 20,   # 20분
            'DOT': 25,   # 25분
            'LINK': 20,  # 20분
            'LTC': 30,   # 30분
            'BCH': 30    # 30분
        }
        
        additional_time = additional_times.get(symbol, 30)
        total_time = base_processing_time + additional_time
        
        # 금액에 따른 추가 시간
        if amount > 10000:  # 1만 달러 이상
            total_time += 30
        elif amount > 1000:  # 1천 달러 이상
            total_time += 15
            
        estimated_completion = datetime.now() + timedelta(minutes=total_time)
        
        return {
            'symbol': symbol,
            'amount': amount,
            'estimated_minutes': total_time,
            'estimated_completion': estimated_completion,
            'confidence': 'HIGH' if total_time < 60 else 'MEDIUM'
        }
        
    def validate_wallet_address(self, symbol: str, address: str) -> bool:
        """지갑 주소 유효성 검사"""
        
        # 기본적인 주소 형식 검사
        if not address or len(address) < 10:
            return False
            
        # 코인별 주소 형식 검사
        if symbol == 'BTC':
            return address.startswith(('1', '3', 'bc1'))
        elif symbol == 'ETH':
            return address.startswith('0x') and len(address) == 42
        elif symbol in ['USDT', 'USDC']:
            return address.startswith('0x') and len(address) == 42  # ERC-20
        elif symbol == 'BNB':
            return address.startswith('bnb1') or (address.startswith('0x') and len(address) == 42)
        elif symbol == 'ADA':
            return address.startswith('addr1')
        elif symbol == 'DOT':
            return address.startswith(('1', '3', '5'))
        elif symbol == 'LINK':
            return address.startswith('0x') and len(address) == 42
        elif symbol in ['LTC', 'BCH']:
            return address.startswith(('L', 'M', '3')) or address.startswith('bitcoincash:')
            
        return True  # 알 수 없는 코인은 기본적으로 허용 