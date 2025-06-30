#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaGenesis 실시간 대시보드 실행 스크립트
"""

import os
import sys
import subprocess
import time
import threading
import webbrowser

def check_dependencies():
    """필요한 패키지가 설치되어 있는지 확인"""
    try:
        import flask
        import flask_cors
        print("✅ Flask 패키지 확인 완료")
    except ImportError:
        print("❌ Flask 패키지가 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요: pip install flask flask-cors")
        return False
    return True

def start_dashboard():
    """대시보드 서버 시작"""
    if not check_dependencies():
        return
    
    print("🚀 AlphaGenesis 실시간 대시보드를 시작합니다...")
    print("=" * 60)
    print("📊 로컬 주소: http://localhost:5001")
    print("🌐 외부 주소: http://34.47.77.230:5001")
    print("🔄 실시간 업데이트: 1초마다")
    print("🎯 백테스트 연동: 자동")
    print("=" * 60)
    
    # 5초 후 브라우저 자동 열기
    def open_browser():
        time.sleep(5)
        try:
            webbrowser.open('http://localhost:5001')
            print("🌐 브라우저가 자동으로 열렸습니다!")
        except:
            print("🌐 브라우저를 수동으로 열어주세요: http://localhost:5001")
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # 대시보드 서버 실행
    try:
        from dashboard.simple_dashboard import app
        app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
    except Exception as e:
        print(f"❌ 대시보드 실행 오류: {e}")
        print("수동으로 실행하세요: python dashboard/simple_dashboard.py")

if __name__ == '__main__':
    start_dashboard() 