#!/usr/bin/env python3
"""
AlphaGenesis 시스템 통합 런처
웹 대시보드, 실전매매, 백테스트 시스템 통합 실행
"""

import sys
import os
import asyncio
import logging
import signal
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 프로젝트 모듈
from dashboard.app import app
from core.live_trading_engine import LiveTradingEngine
from exchange.binance_futures_api import BinanceFuturesAPI
from ml.optimization.parameter_optimizer import ParameterOptimizer
from config.unified_config import config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlphaGenesisSystem:
    """AlphaGenesis 통합 시스템"""
    
    def __init__(self):
        """초기화"""
        self.project_root = Path(__file__).parent
        self.config = config
        
        # 시스템 컴포넌트
        self.web_server = None
        self.trading_engine = None
        self.binance_api = None
        self.optimizer = None
        
        # 시스템 상태
        self.is_running = False
        self.components_status = {
            'web_dashboard': False,
            'trading_engine': False,
            'binance_api': False,
            'ml_optimizer': False
        }
        
        # 실행기
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # 로그 디렉토리 생성
        self.log_dir = self.project_root / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        logger.info("AlphaGenesis 시스템 초기화 완료")
    
    def print_banner(self):
        """시스템 배너 출력"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    █████╗ ██╗     ██████╗ ██╗  ██╗ █████╗  ██████╗ ███████╗███╗   ██╗      ║
║   ██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗██╔════╝ ██╔════╝████╗  ██║      ║
║   ███████║██║     ██████╔╝███████║███████║██║  ███╗█████╗  ██╔██╗ ██║      ║
║   ██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║██║   ██║██╔══╝  ██║╚██╗██║      ║
║   ██║  ██║███████╗██║     ██║  ██║██║  ██║╚██████╔╝███████╗██║ ╚████║      ║
║   ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝      ║
║                                                                              ║
║                    🚀 AI 기반 암호화폐 트레이딩 시스템 v3.0                    ║
║                                                                              ║
║  🌐 웹 대시보드: http://localhost:9000                                        ║
║  📊 백테스트: ML 최적화 지원                                                   ║
║  🔥 실전매매: 24/7 자동 트레이딩                                               ║
║  🎯 바이낸스 선물: 모든 USDT.P 심볼 지원                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    async def initialize_components(self):
        """시스템 컴포넌트 초기화"""
        try:
            logger.info("시스템 컴포넌트 초기화 시작...")
            
            # 1. 바이낸스 API 초기화
            await self.initialize_binance_api()
            
            # 2. ML 최적화 시스템 초기화
            await self.initialize_ml_optimizer()
            
            # 3. 실전매매 엔진 초기화
            await self.initialize_trading_engine()
            
            # 4. 웹 대시보드 초기화
            await self.initialize_web_dashboard()
            
            logger.info("모든 컴포넌트 초기화 완료")
            
        except Exception as e:
            logger.error(f"컴포넌트 초기화 실패: {e}")
            raise
    
    async def initialize_binance_api(self):
        """바이낸스 API 초기화"""
        try:
            logger.info("바이낸스 API 초기화 중...")
            
            # API 키 설정 (환경변수 또는 설정파일에서 로드)
            api_key = os.getenv('BINANCE_API_KEY', '')
            api_secret = os.getenv('BINANCE_API_SECRET', '')
            
            self.binance_api = BinanceFuturesAPI(api_key, api_secret, testnet=True)
            
            # 연결 테스트
            symbols = await self.binance_api.get_usdt_perpetual_symbols()
            logger.info(f"바이낸스 API 연결 성공: {len(symbols)}개 심볼")
            
            self.components_status['binance_api'] = True
            
        except Exception as e:
            logger.error(f"바이낸스 API 초기화 실패: {e}")
            self.components_status['binance_api'] = False
    
    async def initialize_ml_optimizer(self):
        """ML 최적화 시스템 초기화"""
        try:
            logger.info("ML 최적화 시스템 초기화 중...")
            
            self.optimizer = ParameterOptimizer()
            
            # 테스트 최적화 실행
            # test_symbols = ['BTC/USDT']
            # result = await self.optimizer.optimize_parameters(test_symbols, 'triple_combo', n_trials=5)
            # logger.info(f"ML 최적화 테스트 완료: {result.best_score:.4f}")
            
            self.components_status['ml_optimizer'] = True
            
        except Exception as e:
            logger.error(f"ML 최적화 시스템 초기화 실패: {e}")
            self.components_status['ml_optimizer'] = False
    
    async def initialize_trading_engine(self):
        """실전매매 엔진 초기화"""
        try:
            logger.info("실전매매 엔진 초기화 중...")
            
            # API 키 설정
            api_key = os.getenv('BINANCE_API_KEY', '')
            api_secret = os.getenv('BINANCE_API_SECRET', '')
            
            self.trading_engine = LiveTradingEngine(api_key, api_secret, testnet=True)
            await self.trading_engine.initialize()
            
            self.components_status['trading_engine'] = True
            
        except Exception as e:
            logger.error(f"실전매매 엔진 초기화 실패: {e}")
            self.components_status['trading_engine'] = False
    
    async def initialize_web_dashboard(self):
        """웹 대시보드 초기화"""
        try:
            logger.info("웹 대시보드 초기화 중...")
            
            # Flask 앱 설정
            app.config['DEBUG'] = False
            app.config['TESTING'] = False
            
            self.components_status['web_dashboard'] = True
            
        except Exception as e:
            logger.error(f"웹 대시보드 초기화 실패: {e}")
            self.components_status['web_dashboard'] = False
    
    def start_web_server(self):
        """웹 서버 시작"""
        try:
            logger.info("웹 서버 시작 중...")
            
            # Flask 앱 실행
            app.run(
                host='0.0.0.0',
                port=9000,
                debug=False,
                use_reloader=False,
                threaded=True
            )
            
        except Exception as e:
            logger.error(f"웹 서버 시작 실패: {e}")
    
    async def start_system(self, mode: str = 'full'):
        """시스템 시작"""
        try:
            logger.info(f"AlphaGenesis 시스템 시작 (모드: {mode})")
            
            # 컴포넌트 초기화
            await self.initialize_components()
            
            # 시스템 상태 출력
            self.print_system_status()
            
            if mode == 'full':
                # 전체 시스템 실행
                await self.run_full_system()
            elif mode == 'dashboard':
                # 대시보드만 실행
                await self.run_dashboard_only()
            elif mode == 'trading':
                # 실전매매만 실행
                await self.run_trading_only()
            elif mode == 'backtest':
                # 백테스트만 실행
                await self.run_backtest_only()
            
            self.is_running = True
            
        except Exception as e:
            logger.error(f"시스템 시작 실패: {e}")
            raise
    
    async def run_full_system(self):
        """전체 시스템 실행"""
        try:
            logger.info("전체 시스템 모드 실행")
            
            # 웹 서버를 백그라운드에서 실행
            web_server_task = asyncio.create_task(
                asyncio.to_thread(self.start_web_server)
            )
            
            # 메인 루프 실행
            await self.main_loop()
            
        except Exception as e:
            logger.error(f"전체 시스템 실행 실패: {e}")
    
    async def run_dashboard_only(self):
        """대시보드만 실행"""
        try:
            logger.info("대시보드 전용 모드 실행")
            
            # 웹 서버 실행
            self.start_web_server()
            
        except Exception as e:
            logger.error(f"대시보드 실행 실패: {e}")
    
    async def run_trading_only(self):
        """실전매매만 실행"""
        try:
            logger.info("실전매매 전용 모드 실행")
            
            if self.trading_engine:
                # 실전매매 시작
                symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
                await self.trading_engine.start_trading(symbols)
                
                # 실전매매 모니터링
                await self.monitor_trading()
            
        except Exception as e:
            logger.error(f"실전매매 실행 실패: {e}")
    
    async def run_backtest_only(self):
        """백테스트만 실행"""
        try:
            logger.info("백테스트 전용 모드 실행")
            
            if self.optimizer:
                # 백테스트 실행
                symbols = ['BTC/USDT', 'ETH/USDT']
                result = await self.optimizer.optimize_parameters(
                    symbols, 'triple_combo', n_trials=50
                )
                
                logger.info(f"백테스트 완료: {result.best_score:.4f}")
            
        except Exception as e:
            logger.error(f"백테스트 실행 실패: {e}")
    
    async def main_loop(self):
        """메인 루프"""
        try:
            logger.info("메인 루프 시작")
            
            while self.is_running:
                try:
                    # 시스템 상태 모니터링
                    await self.monitor_system()
                    
                    # 30초 대기
                    await asyncio.sleep(30)
                    
                except KeyboardInterrupt:
                    logger.info("사용자 중단 요청")
                    break
                except Exception as e:
                    logger.error(f"메인 루프 오류: {e}")
                    await asyncio.sleep(5)
            
            logger.info("메인 루프 종료")
            
        except Exception as e:
            logger.error(f"메인 루프 실행 실패: {e}")
    
    async def monitor_system(self):
        """시스템 모니터링"""
        try:
            # 컴포넌트 상태 확인
            if self.trading_engine:
                status = await self.trading_engine.get_status()
                logger.info(f"거래 상태: {status.get('status', 'unknown')}")
            
            # 시스템 리소스 모니터링
            # (CPU, 메모리 사용량 등)
            
        except Exception as e:
            logger.error(f"시스템 모니터링 실패: {e}")
    
    async def monitor_trading(self):
        """실전매매 모니터링"""
        try:
            while self.is_running:
                if self.trading_engine:
                    status = await self.trading_engine.get_status()
                    
                    if status.get('status') == 'error':
                        logger.error("실전매매 시스템 오류 발생")
                        break
                
                await asyncio.sleep(60)  # 1분마다 확인
                
        except Exception as e:
            logger.error(f"실전매매 모니터링 실패: {e}")
    
    def print_system_status(self):
        """시스템 상태 출력"""
        print("\n📊 시스템 상태:")
        print("-" * 40)
        
        for component, status in self.components_status.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {component}: {'활성' if status else '비활성'}")
        
        print("-" * 40)
        print(f"🌐 웹 대시보드: http://localhost:9000")
        print(f"📝 로그 파일: {self.log_dir}/system.log")
        print()
    
    def setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            logger.info(f"시그널 {signum} 수신 - 시스템 종료 중...")
            self.is_running = False
            
            # 컴포넌트 정리
            if self.trading_engine:
                asyncio.create_task(self.trading_engine.stop_trading())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """시스템 종료"""
        try:
            logger.info("시스템 종료 중...")
            
            # 실전매매 중지
            if self.trading_engine:
                await self.trading_engine.stop_trading()
            
            # 실행기 종료
            self.executor.shutdown(wait=True)
            
            logger.info("시스템 종료 완료")
            
        except Exception as e:
            logger.error(f"시스템 종료 실패: {e}")

async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="AlphaGenesis 트레이딩 시스템")
    parser.add_argument(
        'mode', 
        choices=['full', 'dashboard', 'trading', 'backtest'],
        default='full',
        nargs='?',
        help='실행 모드'
    )
    parser.add_argument('--version', action='version', version='AlphaGenesis v3.0')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    
    args = parser.parse_args()
    
    # 디버그 모드 설정
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 시스템 생성
    system = AlphaGenesisSystem()
    
    try:
        # 배너 출력
        system.print_banner()
        
        # 시그널 핸들러 설정
        system.setup_signal_handlers()
        
        # 시스템 시작
        await system.start_system(args.mode)
        
    except KeyboardInterrupt:
        logger.info("사용자 중단")
    except Exception as e:
        logger.error(f"시스템 실행 실패: {e}")
        sys.exit(1)
    finally:
        # 시스템 종료
        await system.shutdown()

if __name__ == "__main__":
    # 비동기 메인 함수 실행
    asyncio.run(main())