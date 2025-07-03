import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
import json
import logging
from enum import Enum
import time

class RiskLevel(Enum):
    """위험 수준"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ExchangeBankruptcyDetector:
    """
    거래소 파산 감지 시스템
    - 과거 거래소 파산 사례 분석
    - 실시간 위험 지표 모니터링
    - 텔레그램 경고 시스템
    - 자동 출금 시스템 연동
    """
    
    def __init__(self, telegram_bot_token: str = None, telegram_chat_id: str = None):
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        
        # 과거 거래소 파산 사례 데이터
        self.bankruptcy_cases = self._load_bankruptcy_cases()
        
        # 위험 지표 임계값
        self.risk_thresholds = {
            'volume_drop': 0.7,      # 거래량 70% 감소
            'withdrawal_delay': 24,   # 출금 지연 24시간
            'price_manipulation': 0.3, # 가격 조작 의심 30%
            'liquidity_ratio': 0.1,   # 유동성 비율 10% 미만
            'user_complaints': 100,   # 사용자 불만 100건 이상
            'regulatory_action': True, # 규제 조치
            'executive_exit': True,    # 임원 이탈
            'funding_issues': True     # 자금 조달 문제
        }
        
        # 모니터링 중인 거래소
        self.monitored_exchanges = {
            'binance': {'risk_level': RiskLevel.LOW, 'last_check': None},
            'coinbase': {'risk_level': RiskLevel.LOW, 'last_check': None},
            'kraken': {'risk_level': RiskLevel.LOW, 'last_check': None},
            'kucoin': {'risk_level': RiskLevel.LOW, 'last_check': None},
            'bybit': {'risk_level': RiskLevel.LOW, 'last_check': None},
            'okx': {'risk_level': RiskLevel.LOW, 'last_check': None},
            'gate': {'risk_level': RiskLevel.LOW, 'last_check': None},
            'mexc': {'risk_level': RiskLevel.LOW, 'last_check': None}
        }
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _load_bankruptcy_cases(self) -> List[Dict]:
        """과거 거래소 파산 사례 로드"""
        return [
            {
                'name': 'FTX',
                'date': '2022-11-11',
                'warning_signs': [
                    '거래량 급감',
                    '출금 지연',
                    '가격 조작 의심',
                    '유동성 부족',
                    '규제 조치',
                    '임원 이탈',
                    '자금 조달 문제'
                ],
                'timeline': {
                    'first_warning': '2022-10-01',
                    'critical_warning': '2022-11-01',
                    'bankruptcy': '2022-11-11'
                }
            },
            {
                'name': 'Celsius',
                'date': '2022-07-13',
                'warning_signs': [
                    '출금 중단',
                    '유동성 부족',
                    '사용자 불만 급증',
                    '규제 압박'
                ],
                'timeline': {
                    'first_warning': '2022-06-01',
                    'critical_warning': '2022-07-01',
                    'bankruptcy': '2022-07-13'
                }
            },
            {
                'name': 'Voyager Digital',
                'date': '2022-07-05',
                'warning_signs': [
                    '출금 제한',
                    '유동성 위기',
                    '대출 상환 지연'
                ],
                'timeline': {
                    'first_warning': '2022-06-15',
                    'critical_warning': '2022-06-30',
                    'bankruptcy': '2022-07-05'
                }
            },
            {
                'name': 'Three Arrows Capital',
                'date': '2022-07-01',
                'warning_signs': [
                    '레버리지 포지션 손실',
                    '유동성 부족',
                    '대출 상환 불가'
                ],
                'timeline': {
                    'first_warning': '2022-06-01',
                    'critical_warning': '2022-06-20',
                    'bankruptcy': '2022-07-01'
                }
            }
        ]
        
    def analyze_exchange_health(self, exchange_name: str) -> Dict:
        """거래소 건강도 분석"""
        
        try:
            # 1. 거래량 분석
            volume_analysis = self._analyze_volume_trends(exchange_name)
            
            # 2. 출금 상태 분석
            withdrawal_analysis = self._analyze_withdrawal_status(exchange_name)
            
            # 3. 유동성 분석
            liquidity_analysis = self._analyze_liquidity(exchange_name)
            
            # 4. 사용자 불만 분석
            complaint_analysis = self._analyze_user_complaints(exchange_name)
            
            # 5. 규제 상태 분석
            regulatory_analysis = self._analyze_regulatory_status(exchange_name)
            
            # 6. 뉴스 및 소셜 미디어 분석
            news_analysis = self._analyze_news_sentiment(exchange_name)
            
            # 종합 위험도 계산
            risk_score = self._calculate_risk_score([
                volume_analysis,
                withdrawal_analysis,
                liquidity_analysis,
                complaint_analysis,
                regulatory_analysis,
                news_analysis
            ])
            
            # 위험 수준 결정
            risk_level = self._determine_risk_level(risk_score)
            
            analysis_result = {
                'exchange': exchange_name,
                'timestamp': datetime.now(),
                'risk_score': risk_score,
                'risk_level': risk_level,
                'volume_analysis': volume_analysis,
                'withdrawal_analysis': withdrawal_analysis,
                'liquidity_analysis': liquidity_analysis,
                'complaint_analysis': complaint_analysis,
                'regulatory_analysis': regulatory_analysis,
                'news_analysis': news_analysis,
                'recommendation': self._get_recommendation(risk_level)
            }
            
            # 위험 수준 업데이트
            self.monitored_exchanges[exchange_name]['risk_level'] = risk_level
            self.monitored_exchanges[exchange_name]['last_check'] = datetime.now()
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"거래소 분석 중 오류 발생: {exchange_name} - {str(e)}")
            return {
                'exchange': exchange_name,
                'timestamp': datetime.now(),
                'risk_score': 0.5,
                'risk_level': RiskLevel.MEDIUM,
                'error': str(e),
                'recommendation': '분석 실패 - 주의 필요'
            }
            
    def _analyze_volume_trends(self, exchange_name: str) -> Dict:
        """거래량 트렌드 분석"""
        # 실제 구현에서는 거래소 API를 통해 거래량 데이터 수집
        # 여기서는 시뮬레이션 데이터 사용
        
        try:
            # 24시간 거래량 변화율 (시뮬레이션)
            volume_change_24h = np.random.normal(0, 0.2)  # 평균 0%, 표준편차 20%
            
            # 7일 거래량 변화율
            volume_change_7d = np.random.normal(-0.1, 0.3)  # 평균 -10%, 표준편차 30%
            
            # 거래량 급감 여부
            volume_drop_severe = abs(volume_change_24h) > self.risk_thresholds['volume_drop']
            
            return {
                'volume_change_24h': volume_change_24h,
                'volume_change_7d': volume_change_7d,
                'volume_drop_severe': volume_drop_severe,
                'risk_score': min(abs(volume_change_24h) / self.risk_thresholds['volume_drop'], 1.0)
            }
            
        except Exception as e:
            return {'error': str(e), 'risk_score': 0.5}
            
    def _analyze_withdrawal_status(self, exchange_name: str) -> Dict:
        """출금 상태 분석"""
        try:
            # 출금 지연 시간 (시뮬레이션)
            withdrawal_delay_hours = np.random.exponential(2)  # 평균 2시간
            
            # 출금 실패율
            withdrawal_failure_rate = np.random.beta(1, 10)  # 평균 9% 실패율
            
            # 출금 중단 여부
            withdrawal_suspended = withdrawal_delay_hours > self.risk_thresholds['withdrawal_delay']
            
            return {
                'withdrawal_delay_hours': withdrawal_delay_hours,
                'withdrawal_failure_rate': withdrawal_failure_rate,
                'withdrawal_suspended': withdrawal_suspended,
                'risk_score': min(withdrawal_delay_hours / self.risk_thresholds['withdrawal_delay'], 1.0)
            }
            
        except Exception as e:
            return {'error': str(e), 'risk_score': 0.5}
            
    def _analyze_liquidity(self, exchange_name: str) -> Dict:
        """유동성 분석"""
        try:
            # 유동성 비율 (시뮬레이션)
            liquidity_ratio = np.random.beta(5, 5)  # 평균 50%
            
            # 유동성 부족 여부
            liquidity_insufficient = liquidity_ratio < self.risk_thresholds['liquidity_ratio']
            
            return {
                'liquidity_ratio': liquidity_ratio,
                'liquidity_insufficient': liquidity_insufficient,
                'risk_score': max(0, (self.risk_thresholds['liquidity_ratio'] - liquidity_ratio) / self.risk_thresholds['liquidity_ratio'])
            }
            
        except Exception as e:
            return {'error': str(e), 'risk_score': 0.5}
            
    def _analyze_user_complaints(self, exchange_name: str) -> Dict:
        """사용자 불만 분석"""
        try:
            # 24시간 내 불만 건수 (시뮬레이션)
            complaints_24h = np.random.poisson(20)  # 평균 20건
            
            # 불만 급증 여부
            complaints_surge = complaints_24h > self.risk_thresholds['user_complaints']
            
            return {
                'complaints_24h': complaints_24h,
                'complaints_surge': complaints_surge,
                'risk_score': min(complaints_24h / self.risk_thresholds['user_complaints'], 1.0)
            }
            
        except Exception as e:
            return {'error': str(e), 'risk_score': 0.5}
            
    def _analyze_regulatory_status(self, exchange_name: str) -> Dict:
        """규제 상태 분석"""
        try:
            # 규제 조치 여부 (시뮬레이션)
            regulatory_action = np.random.random() < 0.05  # 5% 확률로 규제 조치
            
            # 규제 압박 수준
            regulatory_pressure = np.random.beta(1, 10) if regulatory_action else 0
            
            return {
                'regulatory_action': regulatory_action,
                'regulatory_pressure': regulatory_pressure,
                'risk_score': 1.0 if regulatory_action else 0.0
            }
            
        except Exception as e:
            return {'error': str(e), 'risk_score': 0.5}
            
    def _analyze_news_sentiment(self, exchange_name: str) -> Dict:
        """뉴스 감정 분석"""
        try:
            # 뉴스 감정 점수 (시뮬레이션)
            sentiment_score = np.random.normal(0, 0.3)  # 평균 0, 표준편차 0.3
            
            # 부정적 뉴스 비율
            negative_news_ratio = np.random.beta(2, 8)  # 평균 20%
            
            # 부정적 뉴스 급증 여부
            negative_surge = negative_news_ratio > 0.5
            
            return {
                'sentiment_score': sentiment_score,
                'negative_news_ratio': negative_news_ratio,
                'negative_surge': negative_surge,
                'risk_score': max(0, negative_news_ratio)
            }
            
        except Exception as e:
            return {'error': str(e), 'risk_score': 0.5}
            
    def _calculate_risk_score(self, analyses: List[Dict]) -> float:
        """종합 위험도 계산"""
        risk_scores = []
        
        for analysis in analyses:
            if 'risk_score' in analysis and not isinstance(analysis['risk_score'], str):
                risk_scores.append(analysis['risk_score'])
                
        if not risk_scores:
            return 0.5
            
        # 가중 평균 계산 (거래량과 출금 상태에 더 높은 가중치)
        weights = [0.25, 0.25, 0.15, 0.15, 0.1, 0.1]  # 총합 1.0
        
        weighted_score = sum(score * weight for score, weight in zip(risk_scores[:len(weights)], weights))
        
        return min(weighted_score, 1.0)
        
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """위험 수준 결정"""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
            
    def _get_recommendation(self, risk_level: RiskLevel) -> str:
        """위험 수준별 권고사항"""
        recommendations = {
            RiskLevel.LOW: "정상 상태 - 계속 모니터링",
            RiskLevel.MEDIUM: "주의 필요 - 출금 준비 권고",
            RiskLevel.HIGH: "위험 상태 - 즉시 출금 권고",
            RiskLevel.CRITICAL: "매우 위험 - 긴급 출금 필수"
        }
        
        return recommendations.get(risk_level, "분석 불가")
        
    def send_telegram_alert(self, analysis_result: Dict):
        """텔레그램 경고 전송"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            self.logger.warning("텔레그램 설정이 없습니다.")
            return
            
        try:
            exchange = analysis_result['exchange']
            risk_level = analysis_result['risk_level'].value
            risk_score = analysis_result['risk_score']
            recommendation = analysis_result['recommendation']
            
            message = f"""
🚨 거래소 위험 경고 🚨

📊 거래소: {exchange.upper()}
⚠️ 위험 수준: {risk_level}
📈 위험 점수: {risk_score:.2f}
💡 권고사항: {recommendation}

⏰ 감지 시간: {analysis_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

🔍 상세 분석:
• 거래량 변화: {analysis_result.get('volume_analysis', {}).get('volume_change_24h', 0):.2%}
• 출금 지연: {analysis_result.get('withdrawal_analysis', {}).get('withdrawal_delay_hours', 0):.1f}시간
• 유동성 비율: {analysis_result.get('liquidity_analysis', {}).get('liquidity_ratio', 0):.2%}
• 사용자 불만: {analysis_result.get('complaint_analysis', {}).get('complaints_24h', 0)}건

⚠️ 즉시 대응 조치를 권고합니다!
            """
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                self.logger.info(f"텔레그램 경고 전송 완료: {exchange}")
            else:
                self.logger.error(f"텔레그램 전송 실패: {response.text}")
                
        except Exception as e:
            self.logger.error(f"텔레그램 전송 중 오류: {str(e)}")
            
    def monitor_all_exchanges(self):
        """모든 거래소 모니터링"""
        self.logger.info("거래소 모니터링 시작...")
        
        for exchange_name in self.monitored_exchanges.keys():
            try:
                analysis_result = self.analyze_exchange_health(exchange_name)
                
                # 위험 수준이 MEDIUM 이상이면 경고
                if analysis_result['risk_level'] in [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    self.send_telegram_alert(analysis_result)
                    
                # 로그 기록
                self.logger.info(f"{exchange_name}: {analysis_result['risk_level'].value} ({analysis_result['risk_score']:.2f})")
                
                # 분석 간격 조절
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"{exchange_name} 모니터링 중 오류: {str(e)}")
                
    def get_exchange_status_summary(self) -> Dict:
        """거래소 상태 요약"""
        summary = {
            'total_exchanges': len(self.monitored_exchanges),
            'risk_distribution': {
                'LOW': 0,
                'MEDIUM': 0,
                'HIGH': 0,
                'CRITICAL': 0
            },
            'exchanges_by_risk': {
                'LOW': [],
                'MEDIUM': [],
                'HIGH': [],
                'CRITICAL': []
            }
        }
        
        for exchange_name, info in self.monitored_exchanges.items():
            risk_level = info['risk_level'].value
            summary['risk_distribution'][risk_level] += 1
            summary['exchanges_by_risk'][risk_level].append(exchange_name)
            
        return summary 