from flask import Blueprint, jsonify, request, Response, render_template, redirect
from .services import dashboard_manager
import json
import queue

# Flask Blueprint 생성
api = Blueprint('api', __name__)

@api.route('/')
def root_redirect():
    return redirect('/backtest')

@api.route('/backtest')
def backtest_dashboard():
    """백테스트 대시보드 페이지 렌더링"""
    return render_template('backtest_dashboard.html')

@api.route('/api/data/info', methods=['GET'])
def get_data_info():
    """데이터 정보 반환 API"""
    info = dashboard_manager.get_data_info()
    if 'error' in info:
        return jsonify(info), 500
    return jsonify(info)

@api.route('/api/data/download', methods=['GET'])
def download_data():
    """데이터 다운로드 API (BTC_USDT 고정)"""
    symbol = 'BTC_USDT'
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    if not start_date or not end_date:
        return jsonify({'error': '시작일과 종료일을 지정해주세요.'}), 400
    
    result = dashboard_manager.download_data(symbol, start_date, end_date)
    if 'error' in result:
        return jsonify(result), 500
    return jsonify(result)

@api.route('/api/strategies', methods=['GET'])
def get_strategies():
    """전략 목록 조회 API"""
    strategies = dashboard_manager.get_strategies()
    if 'error' in strategies:
        return jsonify(strategies), 500
    return jsonify({'strategies': strategies})

@api.route('/api/backtest/run', methods=['POST'])
def run_backtest():
    """백테스트 실행 API"""
    data = request.get_json()
    if not data:
        return jsonify({'error': '요청 데이터가 없습니다.'}), 400
    
    result = dashboard_manager.start_backtest(data)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)

@api.route('/api/backtest/stop', methods=['POST'])
def stop_backtest():
    """백테스트 중지 API"""
    result = dashboard_manager.stop_backtest()
    return jsonify(result)

@api.route('/api/backtest/stream_log')
def stream_backtest_log():
    """백테스트 로그 실시간 스트리밍 (SSE)"""
    def event_stream():
        q = dashboard_manager.backtest_log_queue
        while True:
            try:
                log = q.get(timeout=1)
                if log == "[END]":
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    break
                yield f"data: {log}\n\n"
            except queue.Empty:
                if not dashboard_manager.is_backtest_running:
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    break
                # 연결 유지를 위한 ping
                yield ": ping\n\n"
                
    return Response(event_stream(), mimetype="text/event-stream")

@api.route('/api/backtest/results')
def get_backtest_results():
    """최신 백테스트 결과 조회 API"""
    results = dashboard_manager.get_latest_results()
    if 'error' in results:
        return jsonify(results), 404
    return jsonify(results)
