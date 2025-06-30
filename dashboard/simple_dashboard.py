#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import os
from datetime import datetime
import threading
import time
import re

app = Flask(__name__)
CORS(app)

# 실시간 데이터 저장소
realtime_data = {
    'current_capital': 10000000,
    'total_return': 0.0,
    'realized_pnl': 0,
    'unrealized_pnl': 0,
    'open_positions': 0,
    'ml_prediction': 0.0,
    'strategy': '크로노스',
    'regime': '횡보',
    'trades_count': 0,
    'win_rate': 0.0,
    'logs': [],
    'capital_history': [10000000],
    'timestamp_history': [datetime.now().strftime('%H:%M:%S')],
    'last_update': datetime.now()
}

@app.route('/')
def index():
    """메인 대시보드 페이지"""
    return render_template('realtime_dashboard.html')

@app.route('/api/realtime_data')
def get_realtime_data():
    """실시간 데이터 API"""
    return jsonify(realtime_data)

@app.route('/api/realtime_log', methods=['POST'])
def receive_log():
    """백테스트에서 전송되는 실시간 로그 수신"""
    try:
        data = request.get_json()
        log_msg = data.get('log', '')
        
        # 디버깅: 받은 로그 출력
        print(f"[DEBUG] 받은 로그: {log_msg}")
        
        # 로그 저장 (최근 100개만 유지)
        if len(realtime_data['logs']) >= 100:
            realtime_data['logs'].pop(0)
        realtime_data['logs'].append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'message': log_msg
        })
        
        # 데이터 파싱 및 업데이트
        parse_and_update_data(log_msg)
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"로그 수신 오류: {e}")
        return jsonify({'error': str(e)}), 500

def parse_and_update_data(log_msg):
    """로그 메시지에서 데이터를 파싱하여 실시간 데이터 업데이트"""
    try:
        # 디버깅: 파싱 시도
        print(f"[DEBUG] 파싱 중: {log_msg[:100]}...")
        
        # 총자산 파싱 (더 강력한 정규식)
        capital_match = re.search(r'총자산[:\s]*([+-]?[\d,]+(?:\.\d+)?)', log_msg)
        if capital_match:
            new_capital = float(capital_match.group(1).replace(',', ''))
            realtime_data['current_capital'] = new_capital
            print(f"[DEBUG] 총자산 업데이트: {new_capital:,.0f}")
            
            # 차트용 데이터 추가 (최근 50개만 유지)
            if len(realtime_data['capital_history']) >= 50:
                realtime_data['capital_history'].pop(0)
                realtime_data['timestamp_history'].pop(0)
            realtime_data['capital_history'].append(new_capital)
            realtime_data['timestamp_history'].append(datetime.now().strftime('%H:%M:%S'))
        
        # 수익률 파싱 (더 강력한 정규식)
        return_match = re.search(r'수익률[:\s]*([+-]?[\d.]+)%', log_msg)
        if return_match:
            realtime_data['total_return'] = float(return_match.group(1))
            print(f"[DEBUG] 수익률 업데이트: {return_match.group(1)}%")
        
        # 실현손익 파싱
        realized_match = re.search(r'실현손익[:\s]*([+-]?[\d,]+(?:\.\d+)?)', log_msg)
        if realized_match:
            value = float(realized_match.group(1).replace(',', ''))
            realtime_data['realized_pnl'] = value
            print(f"[DEBUG] 실현손익 업데이트: {value:,.0f}")
        
        # 미실현손익 파싱
        unrealized_match = re.search(r'미실현손익[:\s]*([+-]?[\d,]+(?:\.\d+)?)', log_msg)
        if unrealized_match:
            value = float(unrealized_match.group(1).replace(',', ''))
            realtime_data['unrealized_pnl'] = value
            print(f"[DEBUG] 미실현손익 업데이트: {value:,.0f}")
        
        # 포지션 수 파싱
        position_match = re.search(r'보유포지션[:\s]*(\d+)개?', log_msg)
        if position_match:
            realtime_data['open_positions'] = int(position_match.group(1))
            print(f"[DEBUG] 포지션 업데이트: {position_match.group(1)}개")
        
        # ML 예측값 파싱 (더 넓은 범위)
        ml_match = re.search(r'ML예측[:\s]*([+-]?[\d.]+)%', log_msg)
        if ml_match:
            realtime_data['ml_prediction'] = float(ml_match.group(1))
            print(f"[DEBUG] ML예측 업데이트: {ml_match.group(1)}%")
        
        # 전략 파싱 (순서 중요: 더 구체적인 것부터)
        strategy_patterns = [
            ('비트코인숏전략', '비트코인숏전략'),
            ('모멘텀돌파', '모멘텀돌파'),
            ('숏모멘텀', '숏모멘텀'),
            ('추세추종', '추세추종'),
            ('역추세', '역추세')
        ]
        
        for pattern, strategy_name in strategy_patterns:
            if pattern in log_msg:
                realtime_data['strategy'] = strategy_name
                print(f"[DEBUG] 전략 업데이트: {strategy_name}")
                break
        
        # 시장국면 파싱
        regime_patterns = [
            ('급등', '급등'),
            ('급락', '급락'),
            ('상승', '상승'),
            ('하락', '하락'),
            ('횡보', '횡보')
        ]
        
        for pattern, regime_name in regime_patterns:
            if pattern in log_msg:
                realtime_data['regime'] = regime_name
                print(f"[DEBUG] 시장국면 업데이트: {regime_name}")
                break
        
        # 거래 발생시 카운트 증가
        if '진입' in log_msg or '청산' in log_msg:
            realtime_data['trades_count'] += 1
            print(f"[DEBUG] 거래 카운트 증가: {realtime_data['trades_count']}")
        
        realtime_data['last_update'] = datetime.now()
        print(f"[DEBUG] 데이터 업데이트 완료: {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"데이터 파싱 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("🚀 AlphaGenesis 실시간 대시보드 시작!")
    print("📊 로컬 접속: http://localhost:5000")
    print("🌐 외부 접속: http://34.47.77.230:5000")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True) 