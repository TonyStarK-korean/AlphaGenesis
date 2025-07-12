#!/usr/bin/env python3
"""
🚀 성능 및 응답성 최적화 모듈
시스템 리소스 관리, 캐싱, 비동기 처리 최적화
"""

import asyncio
import threading
import time
from functools import wraps, lru_cache
from typing import Dict, Any, Callable, Optional
import logging
import gc
import psutil
import weakref
from datetime import datetime, timedelta
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """
    성능 최적화 관리자
    - 메모리 사용량 최적화
    - API 응답 캐싱
    - 비동기 처리 관리
    - 리소스 모니터링
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.api_call_times = {}
        self.memory_monitor = MemoryMonitor()
        self.response_cache = ResponseCache()
        self.async_pool = AsyncTaskPool()
        
    def cache_response(self, ttl: int = 300):
        """API 응답 캐싱 데코레이터"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 캐시 키 생성
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # 캐시에서 확인
                cached_result = self.response_cache.get(cache_key)
                if cached_result is not None:
                    self.cache_stats['hits'] += 1
                    return cached_result
                
                # 함수 실행
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # 실행 시간 기록
                self.api_call_times[func.__name__] = execution_time
                
                # 결과 캐싱
                self.response_cache.set(cache_key, result, ttl)
                self.cache_stats['misses'] += 1
                
                return result
            return wrapper
        return decorator
    
    def monitor_performance(self, func: Callable) -> Callable:
        """성능 모니터링 데코레이터"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                execution_time = end_time - start_time
                memory_usage = end_memory - start_memory
                
                # 성능 로그
                if execution_time > 1.0:  # 1초 이상 소요시 경고
                    logger.warning(f"느린 함수 실행: {func.__name__} - {execution_time:.2f}초")
                
                if memory_usage > 50:  # 50MB 이상 메모리 사용시 경고
                    logger.warning(f"높은 메모리 사용: {func.__name__} - {memory_usage:.1f}MB")
        
        return wrapper
    
    def async_execute(self, func: Callable, *args, **kwargs):
        """비동기 함수 실행"""
        return self.async_pool.submit(func, *args, **kwargs)
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """캐시 키 생성"""
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        return json.dumps(key_data, sort_keys=True)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 조회"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        
        return {
            'system': {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_info.percent,
                'available_memory': memory_info.available / 1024 / 1024 / 1024,  # GB
                'total_memory': memory_info.total / 1024 / 1024 / 1024  # GB
            },
            'cache': {
                'hit_rate': self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses']) * 100 
                           if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 else 0,
                'total_hits': self.cache_stats['hits'],
                'total_misses': self.cache_stats['misses'],
                'cache_size': len(self.response_cache.cache)
            },
            'api_performance': {
                'slowest_endpoints': sorted(
                    [(endpoint, time_ms) for endpoint, time_ms in self.api_call_times.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                'average_response_time': sum(self.api_call_times.values()) / len(self.api_call_times) 
                                       if self.api_call_times else 0
            },
            'async_tasks': {
                'active_tasks': self.async_pool.active_tasks,
                'completed_tasks': self.async_pool.completed_tasks,
                'failed_tasks': self.async_pool.failed_tasks
            }
        }
    
    def optimize_memory(self):
        """메모리 최적화 실행"""
        # 가비지 컬렉션 강제 실행
        collected = gc.collect()
        
        # 캐시 정리
        self.response_cache.cleanup_expired()
        
        # 오래된 API 통계 제거
        cutoff_time = time.time() - 3600  # 1시간
        self.api_call_times = {
            k: v for k, v in self.api_call_times.items() 
            if v > cutoff_time
        }
        
        logger.info(f"메모리 최적화 완료: {collected}개 객체 정리")
        
        return {
            'objects_collected': collected,
            'cache_cleaned': True,
            'stats_cleaned': True
        }
    
    def clear_cache(self, pattern: Optional[str] = None):
        """캐시 정리"""
        if pattern:
            # 패턴 매칭으로 선택적 정리
            keys_to_remove = [k for k in self.response_cache.cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.response_cache.cache[key]
        else:
            # 전체 캐시 정리
            self.response_cache.cache.clear()
            self.cache_stats = {'hits': 0, 'misses': 0}
        
        logger.info(f"캐시 정리 완료: {'패턴 ' + pattern if pattern else '전체'}")

class ResponseCache:
    """응답 캐싱 시스템"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        if key in self.cache:
            entry = self.cache[key]
            if entry['expires'] > time.time():
                self.access_times[key] = time.time()
                return entry['data']
            else:
                # 만료된 항목 제거
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """캐시에 값 저장"""
        # 최대 크기 확인
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = {
            'data': value,
            'expires': time.time() + ttl,
            'created': time.time()
        }
        self.access_times[key] = time.time()
    
    def _evict_oldest(self):
        """가장 오래된 항목 제거 (LRU)"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def cleanup_expired(self):
        """만료된 항목 정리"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry['expires'] <= current_time
        ]
        
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
        
        return len(expired_keys)

class MemoryMonitor:
    """메모리 모니터링"""
    
    def __init__(self):
        self.memory_history = []
        self.alerts = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: int = 30):
        """메모리 모니터링 시작"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self.monitor_thread.start()
        logger.info("메모리 모니터링 시작")
    
    def stop_monitoring(self):
        """메모리 모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.info("메모리 모니터링 중지")
    
    def _monitor_loop(self, interval: int):
        """모니터링 루프"""
        while self.monitoring:
            try:
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                
                memory_data = {
                    'timestamp': datetime.now().isoformat(),
                    'memory_percent': memory_info.percent,
                    'cpu_percent': cpu_percent,
                    'available_gb': memory_info.available / 1024 / 1024 / 1024
                }
                
                self.memory_history.append(memory_data)
                
                # 히스토리 크기 제한 (최근 100개만 유지)
                if len(self.memory_history) > 100:
                    self.memory_history = self.memory_history[-100:]
                
                # 알림 조건 확인
                if memory_info.percent > 85:
                    self.alerts.append({
                        'type': 'high_memory',
                        'message': f"높은 메모리 사용률: {memory_info.percent:.1f}%",
                        'timestamp': datetime.now().isoformat()
                    })
                
                if cpu_percent > 90:
                    self.alerts.append({
                        'type': 'high_cpu',
                        'message': f"높은 CPU 사용률: {cpu_percent:.1f}%",
                        'timestamp': datetime.now().isoformat()
                    })
                
                # 알림 히스토리 제한
                if len(self.alerts) > 50:
                    self.alerts = self.alerts[-50:]
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"메모리 모니터링 오류: {e}")
                time.sleep(interval)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 조회"""
        if not self.memory_history:
            return {'error': '모니터링 데이터가 없습니다'}
        
        recent_data = self.memory_history[-10:]  # 최근 10개
        
        return {
            'current': self.memory_history[-1] if self.memory_history else None,
            'average_memory': sum(d['memory_percent'] for d in recent_data) / len(recent_data),
            'average_cpu': sum(d['cpu_percent'] for d in recent_data) / len(recent_data),
            'peak_memory': max(d['memory_percent'] for d in recent_data),
            'peak_cpu': max(d['cpu_percent'] for d in recent_data),
            'alerts': self.alerts[-5:],  # 최근 5개 알림
            'history_count': len(self.memory_history)
        }

class AsyncTaskPool:
    """비동기 작업 풀"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.task_queue = asyncio.Queue()
        self.running = False
    
    def submit(self, func: Callable, *args, **kwargs):
        """비동기 작업 제출"""
        if not self.running:
            self.start()
        
        task = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'submitted_at': time.time()
        }
        
        asyncio.create_task(self.task_queue.put(task))
        return task
    
    def start(self):
        """작업 풀 시작"""
        if self.running:
            return
        
        self.running = True
        for _ in range(self.max_workers):
            asyncio.create_task(self._worker())
        
        logger.info(f"비동기 작업 풀 시작: {self.max_workers}개 워커")
    
    async def _worker(self):
        """워커 루프"""
        while self.running:
            try:
                task = await self.task_queue.get()
                
                self.active_tasks += 1
                
                try:
                    # 동기 함수를 비동기로 실행
                    if asyncio.iscoroutinefunction(task['func']):
                        await task['func'](*task['args'], **task['kwargs'])
                    else:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, task['func'], *task['args'], **task['kwargs'])
                    
                    self.completed_tasks += 1
                    
                except Exception as e:
                    logger.error(f"비동기 작업 실패: {e}")
                    self.failed_tasks += 1
                
                finally:
                    self.active_tasks -= 1
                    self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"워커 오류: {e}")
    
    def stop(self):
        """작업 풀 중지"""
        self.running = False
        logger.info("비동기 작업 풀 중지")

# 전역 최적화 인스턴스
performance_optimizer = PerformanceOptimizer()

def get_performance_optimizer():
    """성능 최적화기 인스턴스 반환"""
    return performance_optimizer

# 편의 함수들
def cache_api_response(ttl: int = 300):
    """API 응답 캐싱 데코레이터"""
    return performance_optimizer.cache_response(ttl)

def monitor_api_performance(func: Callable) -> Callable:
    """API 성능 모니터링 데코레이터"""
    return performance_optimizer.monitor_performance(func)

def optimize_system():
    """시스템 최적화 실행"""
    return performance_optimizer.optimize_memory()

if __name__ == "__main__":
    print("🚀 성능 최적화 시스템")
    
    # 최적화기 테스트
    optimizer = PerformanceOptimizer()
    
    # 메모리 모니터링 시작
    optimizer.memory_monitor.start_monitoring(5)
    
    # 테스트 함수
    @optimizer.cache_response(ttl=60)
    def test_cached_function(x, y):
        time.sleep(0.1)  # 시뮬레이션 지연
        return x + y
    
    # 캐시 테스트
    print("첫 번째 호출:", test_cached_function(1, 2))
    print("두 번째 호출 (캐시됨):", test_cached_function(1, 2))
    
    # 성능 통계
    stats = optimizer.get_performance_stats()
    print("성능 통계:", json.dumps(stats, indent=2, ensure_ascii=False))
    
    # 메모리 최적화
    optimization_result = optimizer.optimize_memory()
    print("최적화 결과:", optimization_result)
    
    # 모니터링 중지
    time.sleep(2)
    optimizer.memory_monitor.stop_monitoring()