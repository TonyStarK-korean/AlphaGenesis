#!/usr/bin/env python3
"""
바이낸스 선물 API 연동 시스템
USDT.P 심볼 지원
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceFuturesAPI:
    """바이낸스 선물 API 클래스"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        """
        초기화
        
        Args:
            api_key: API 키
            api_secret: API 시크릿
            testnet: 테스트넷 사용 여부
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # CCXT 거래소 객체 생성
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': testnet,
            'options': {
                'defaultType': 'future',  # 선물 거래
            }
        })
        
        # 심볼 캐시
        self.symbols_cache = {}
        self.last_symbols_update = 0
        
        # 가격 데이터 캐시
        self.price_cache = {}
        self.cache_timeout = 30  # 30초 캐시
        
        logger.info(f"바이낸스 선물 API 초기화 완료 (테스트넷: {testnet})")
    
    async def get_usdt_perpetual_symbols(self) -> List[str]:
        """
        USDT 무기한 선물 심볼 목록 조회
        
        Returns:
            USDT.P 심볼 리스트
        """
        try:
            # 캐시 확인
            current_time = time.time()
            if (current_time - self.last_symbols_update) < 3600:  # 1시간 캐시
                if self.symbols_cache:
                    return self.symbols_cache.get('usdt_perp', [])
            
            # 마켓 정보 조회
            markets = self.exchange.load_markets()
            
            # USDT 무기한 선물 필터링
            usdt_perp_symbols = []
            for symbol, market in markets.items():
                if (market.get('type') == 'future' and 
                    market.get('settle') == 'USDT' and 
                    market.get('expiry') is None):  # 무기한 선물
                    usdt_perp_symbols.append(symbol)
            
            # 캐시 업데이트
            self.symbols_cache['usdt_perp'] = usdt_perp_symbols
            self.last_symbols_update = current_time
            
            logger.info(f"USDT 무기한 선물 심볼 {len(usdt_perp_symbols)}개 조회 완료")
            return usdt_perp_symbols
            
        except Exception as e:
            logger.error(f"심볼 조회 실패: {e}")
            return []
    
    async def get_top_volume_symbols(self, limit: int = 50) -> List[str]:
        """
        거래량 기준 상위 심볼 조회
        
        Args:
            limit: 조회할 심볼 개수
            
        Returns:
            거래량 상위 심볼 리스트
        """
        try:
            # 24시간 티커 데이터 조회
            tickers = self.exchange.fetch_tickers()
            
            # USDT 무기한 선물만 필터링
            usdt_perp_symbols = await self.get_usdt_perpetual_symbols()
            
            # 거래량 기준 정렬
            volume_data = []
            for symbol in usdt_perp_symbols:
                if symbol in tickers:
                    ticker = tickers[symbol]
                    volume_data.append({
                        'symbol': symbol,
                        'volume': ticker.get('quoteVolume', 0),
                        'price': ticker.get('last', 0),
                        'change': ticker.get('percentage', 0)
                    })
            
            # 거래량 기준 내림차순 정렬
            volume_data.sort(key=lambda x: x['volume'], reverse=True)
            
            # 상위 N개 심볼 반환
            top_symbols = [item['symbol'] for item in volume_data[:limit]]
            
            logger.info(f"거래량 상위 {len(top_symbols)}개 심볼 조회 완료")
            return top_symbols
            
        except Exception as e:
            logger.error(f"상위 거래량 심볼 조회 실패: {e}")
            return []
    
    async def get_market_trending_symbols(self, limit: int = 20) -> List[str]:
        """
        시장 트렌딩 심볼 조회 (변동률 + 거래량 기준)
        
        Args:
            limit: 조회할 심볼 개수
            
        Returns:
            트렌딩 심볼 리스트
        """
        try:
            # 24시간 티커 데이터 조회
            tickers = self.exchange.fetch_tickers()
            
            # USDT 무기한 선물만 필터링
            usdt_perp_symbols = await self.get_usdt_perpetual_symbols()
            
            # 트렌딩 점수 계산
            trending_data = []
            for symbol in usdt_perp_symbols:
                if symbol in tickers:
                    ticker = tickers[symbol]
                    
                    volume = ticker.get('quoteVolume', 0)
                    price_change = abs(ticker.get('percentage', 0))
                    
                    # 트렌딩 점수 = (변동률 * 0.6) + (거래량 정규화 * 0.4)
                    volume_score = min(volume / 1000000, 100)  # 거래량 정규화
                    trending_score = (price_change * 0.6) + (volume_score * 0.4)
                    
                    trending_data.append({
                        'symbol': symbol,
                        'trending_score': trending_score,
                        'volume': volume,
                        'price_change': ticker.get('percentage', 0),
                        'price': ticker.get('last', 0)
                    })
            
            # 트렌딩 점수 기준 내림차순 정렬
            trending_data.sort(key=lambda x: x['trending_score'], reverse=True)
            
            # 상위 N개 심볼 반환
            trending_symbols = [item['symbol'] for item in trending_data[:limit]]
            
            logger.info(f"트렌딩 상위 {len(trending_symbols)}개 심볼 조회 완료")
            return trending_symbols
            
        except Exception as e:
            logger.error(f"트렌딩 심볼 조회 실패: {e}")
            return []
    
    async def get_ohlcv_data(self, symbol: str, timeframe: str = '1h', 
                            limit: int = 1000, start_date: str = None) -> pd.DataFrame:
        """
        OHLCV 데이터 조회
        
        Args:
            symbol: 심볼 (예: 'BTC/USDT')
            timeframe: 시간프레임 ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: 데이터 개수
            start_date: 시작일 (YYYY-MM-DD)
            
        Returns:
            OHLCV 데이터프레임
        """
        try:
            # 캐시 키 생성
            cache_key = f"{symbol}_{timeframe}_{limit}"
            current_time = time.time()
            
            # 캐시 확인
            if cache_key in self.price_cache:
                cache_data = self.price_cache[cache_key]
                if (current_time - cache_data['timestamp']) < self.cache_timeout:
                    logger.info(f"캐시에서 {symbol} 데이터 반환")
                    return cache_data['data']
            
            # 시작 시간 계산
            since = None
            if start_date:
                since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            
            # OHLCV 데이터 조회
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            # 데이터프레임 생성
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 기술적 지표 계산
            df = self.calculate_technical_indicators(df)
            
            # 캐시 저장
            self.price_cache[cache_key] = {
                'data': df,
                'timestamp': current_time
            }
            
            logger.info(f"{symbol} OHLCV 데이터 {len(df)}개 조회 완료")
            return df
            
        except Exception as e:
            logger.error(f"{symbol} OHLCV 데이터 조회 실패: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 계산
        
        Args:
            df: OHLCV 데이터프레임
            
        Returns:
            기술적 지표가 추가된 데이터프레임
        """
        if len(df) < 50:
            return df
        
        try:
            # 이동평균
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # 지수이동평균
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # 볼린저 밴드
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # 거래량 지표
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # 변동성
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            logger.info("기술적 지표 계산 완료")
            
        except Exception as e:
            logger.error(f"기술적 지표 계산 실패: {e}")
        
        return df
    
    async def get_realtime_price(self, symbol: str) -> Dict[str, Any]:
        """
        실시간 가격 조회
        
        Args:
            symbol: 심볼
            
        Returns:
            실시간 가격 정보
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'quote_volume': ticker['quoteVolume'],
                'change': ticker['percentage'],
                'timestamp': ticker['timestamp']
            }
            
        except Exception as e:
            logger.error(f"{symbol} 실시간 가격 조회 실패: {e}")
            return {}
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """
        시장 개요 조회
        
        Returns:
            시장 개요 정보
        """
        try:
            # 주요 심볼들의 티커 데이터 조회
            major_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
            
            overview = {
                'timestamp': datetime.now().isoformat(),
                'major_symbols': {},
                'market_stats': {}
            }
            
            total_volume = 0
            positive_count = 0
            negative_count = 0
            
            for symbol in major_symbols:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    
                    overview['major_symbols'][symbol] = {
                        'price': ticker['last'],
                        'change': ticker['percentage'],
                        'volume': ticker['quoteVolume']
                    }
                    
                    total_volume += ticker['quoteVolume']
                    
                    if ticker['percentage'] > 0:
                        positive_count += 1
                    else:
                        negative_count += 1
                        
                except Exception as e:
                    logger.error(f"{symbol} 티커 조회 실패: {e}")
            
            # 시장 통계
            overview['market_stats'] = {
                'total_volume': total_volume,
                'positive_symbols': positive_count,
                'negative_symbols': negative_count,
                'market_sentiment': 'bullish' if positive_count > negative_count else 'bearish'
            }
            
            return overview
            
        except Exception as e:
            logger.error(f"시장 개요 조회 실패: {e}")
            return {}
    
    async def get_funding_rates(self, symbols: List[str]) -> Dict[str, float]:
        """
        펀딩 비율 조회
        
        Args:
            symbols: 심볼 리스트
            
        Returns:
            심볼별 펀딩 비율
        """
        try:
            funding_rates = {}
            
            for symbol in symbols:
                try:
                    # 펀딩 비율 조회
                    funding = self.exchange.fetch_funding_rate(symbol)
                    funding_rates[symbol] = funding['fundingRate']
                    
                except Exception as e:
                    logger.error(f"{symbol} 펀딩 비율 조회 실패: {e}")
                    funding_rates[symbol] = 0.0
            
            return funding_rates
            
        except Exception as e:
            logger.error(f"펀딩 비율 조회 실패: {e}")
            return {}
    
    async def download_historical_data(self, symbols: List[str], 
                                     timeframe: str = '1h', 
                                     days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        여러 심볼의 과거 데이터 일괄 다운로드
        
        Args:
            symbols: 심볼 리스트
            timeframe: 시간프레임
            days: 다운로드할 일수
            
        Returns:
            심볼별 데이터프레임 딕셔너리
        """
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            data_dict = {}
            
            for symbol in symbols:
                try:
                    df = await self.get_ohlcv_data(symbol, timeframe, limit=days*24, start_date=start_date)
                    
                    if not df.empty:
                        data_dict[symbol] = df
                        logger.info(f"{symbol} 데이터 다운로드 완료: {len(df)}개")
                    else:
                        logger.warning(f"{symbol} 데이터 다운로드 실패")
                        
                except Exception as e:
                    logger.error(f"{symbol} 데이터 다운로드 실패: {e}")
                
                # API 제한 방지를 위한 딜레이
                await asyncio.sleep(0.1)
            
            logger.info(f"총 {len(data_dict)}개 심볼 데이터 다운로드 완료")
            return data_dict
            
        except Exception as e:
            logger.error(f"과거 데이터 다운로드 실패: {e}")
            return {}

# 유틸리티 함수들
async def get_binance_universe(api_key: str = None, api_secret: str = None, 
                              top_count: int = 50) -> List[str]:
    """
    바이낸스 USDT.P 유니버스 조회
    
    Args:
        api_key: API 키
        api_secret: API 시크릿
        top_count: 상위 심볼 개수
        
    Returns:
        선별된 심볼 리스트
    """
    try:
        api = BinanceFuturesAPI(api_key, api_secret)
        
        # 거래량 상위 심볼 조회
        top_symbols = await api.get_top_volume_symbols(top_count)
        
        return top_symbols
        
    except Exception as e:
        logger.error(f"바이낸스 유니버스 조회 실패: {e}")
        return []

async def get_hot_universe(api_key: str = None, api_secret: str = None, 
                          count: int = 20) -> List[str]:
    """
    핫 유니버스 조회 (트렌딩 + 거래량 기준)
    
    Args:
        api_key: API 키
        api_secret: API 시크릿
        count: 조회할 심볼 개수
        
    Returns:
        핫 유니버스 심볼 리스트
    """
    try:
        api = BinanceFuturesAPI(api_key, api_secret)
        
        # 트렌딩 심볼 조회
        hot_symbols = await api.get_market_trending_symbols(count)
        
        return hot_symbols
        
    except Exception as e:
        logger.error(f"핫 유니버스 조회 실패: {e}")
        return []

# 테스트 코드
async def test_binance_api():
    """바이낸스 API 테스트"""
    try:
        print("🚀 바이낸스 선물 API 테스트 시작")
        
        # API 초기화 (키 없이 공개 데이터만 사용)
        api = BinanceFuturesAPI()
        
        # 1. USDT 무기한 선물 심볼 조회
        print("\n📊 USDT 무기한 선물 심볼 조회 중...")
        symbols = await api.get_usdt_perpetual_symbols()
        print(f"✅ 총 {len(symbols)}개 심볼 조회 완료")
        print(f"예시: {symbols[:10]}")
        
        # 2. 거래량 상위 심볼 조회
        print("\n📈 거래량 상위 심볼 조회 중...")
        top_symbols = await api.get_top_volume_symbols(10)
        print(f"✅ 거래량 상위 10개 심볼: {top_symbols}")
        
        # 3. 트렌딩 심볼 조회
        print("\n🔥 트렌딩 심볼 조회 중...")
        trending_symbols = await api.get_market_trending_symbols(10)
        print(f"✅ 트렌딩 상위 10개 심볼: {trending_symbols}")
        
        # 4. BTC/USDT 데이터 조회
        print("\n💰 BTC/USDT 데이터 조회 중...")
        btc_data = await api.get_ohlcv_data('BTC/USDT', '1h', 100)
        print(f"✅ BTC/USDT 데이터 {len(btc_data)}개 조회 완료")
        print(f"최신 가격: {btc_data['close'].iloc[-1]:.2f}")
        
        # 5. 시장 개요 조회
        print("\n🌍 시장 개요 조회 중...")
        market_overview = await api.get_market_overview()
        print(f"✅ 시장 개요 조회 완료")
        print(f"시장 심리: {market_overview.get('market_stats', {}).get('market_sentiment', 'unknown')}")
        
        print("\n🎉 바이낸스 API 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    # 비동기 테스트 실행
    asyncio.run(test_binance_api())