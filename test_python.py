#!/usr/bin/env python3
"""
Python 설치 및 라이브러리 테스트 스크립트
"""

import sys
import os

print("🚀 Python 환경 테스트 시작")
print("=" * 50)

# Python 버전 확인
print(f"📍 Python 버전: {sys.version}")
print(f"📍 Python 경로: {sys.executable}")
print()

# 기본 라이브러리 확인
required_libraries = [
    'os', 'sys', 'datetime', 'json',
    'pandas', 'numpy', 'matplotlib', 
    'sklearn', 'flask', 'tqdm'
]

print("📚 라이브러리 가용성 검사:")
print("-" * 30)

available_count = 0
for lib in required_libraries:
    try:
        __import__(lib)
        print(f"✅ {lib:<15} : 사용 가능")
        available_count += 1
    except ImportError:
        print(f"❌ {lib:<15} : 설치 필요")

print()
print(f"📊 총 {len(required_libraries)}개 중 {available_count}개 사용 가능")

if available_count >= 8:
    print("🎉 백테스트 실행 준비 완료!")
else:
    print("⚠️  추가 라이브러리 설치가 필요합니다.")
    print()
    print("💡 설치 명령어:")
    missing_libs = ['pandas', 'numpy', 'matplotlib', 'scikit-learn', 'flask', 'tqdm']
    print(f"   py -m pip install {' '.join(missing_libs)}")

print()
print("🔧 현재 작업 디렉토리:", os.getcwd())
print("📁 프로젝트 파일 확인:")

# 주요 파일 존재 확인
important_files = [
    'run_ml_backtest.py',
    'run_triple_combo_backtest.py', 
    'triple_combo_strategy.py',
    'dashboard/app.py',
    'config/unified_config.py'
]

for file in important_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} (파일 없음)")

print()
print("✨ 테스트 완료!")