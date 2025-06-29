#!/usr/bin/env python3
"""
데이터 다운로드 실행 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.market_data_downloader import MarketDataDownloader
from config.backtest_config import backtest_config
import logging

def main():
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_download.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("데이터 다운로드 시작")
    
    try:
        # 다운로더 초기화
        downloader = MarketDataDownloader()
        
        # 설정 확인
        config_summary = backtest_config.get_config_summary()
        logger.info(f"다운로드 설정: {config_summary['data_download']}")
        logger.info(f"기간: {config_summary['date_range']}")
        
        # 모든 데이터 다운로드
        all_data = downloader.download_all_data()
        
        # 결과 요약
        logger.info("=== 다운로드 완료 ===")
        for symbol, df in all_data.items():
            if not df.empty:
                logger.info(f"{symbol}: {len(df)}개 레코드 ({df.index[0]} ~ {df.index[-1]})")
            else:
                logger.warning(f"{symbol}: 데이터 없음")
                
        logger.info(f"총 {len(all_data)}개 종목 데이터 다운로드 완료")
        
    except Exception as e:
        logger.error(f"데이터 다운로드 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 