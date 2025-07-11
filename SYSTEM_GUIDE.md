# AlphaGenesis v3.0 시스템 가이드

## 🚀 시스템 개요

AlphaGenesis는 AI 기반 암호화폐 자동매매 시스템으로, 웹 대시보드를 통해 백테스트와 실전매매를 통합 관리할 수 있습니다.

### 주요 기능

- 🌐 **웹 대시보드**: 직관적인 UI로 모든 기능 제어
- 📊 **백테스트**: ML 최적화를 통한 전략 검증
- 🔥 **실전매매**: 24/7 자동 트레이딩 시스템
- 🎯 **바이낸스 선물**: 모든 USDT.P 심볼 지원
- 🧠 **ML 최적화**: 파라미터 자동 최적화
- 🔒 **리스크 관리**: 다층 위험 관리 시스템

## 📁 프로젝트 구조

```
C:\Project\alphagenesis\
├── 📂 dashboard/              # 웹 대시보드
│   ├── 📂 templates/         # HTML 템플릿
│   ├── 📂 static/           # CSS, JS 파일
│   ├── app.py              # Flask 앱
│   └── routes.py           # API 라우트
├── 📂 core/                  # 핵심 엔진
│   ├── live_trading_engine.py  # 실전매매 엔진
│   ├── position_management.py # 포지션 관리
│   └── risk_management.py     # 리스크 관리
├── 📂 exchange/              # 거래소 연동
│   └── binance_futures_api.py # 바이낸스 선물 API
├── 📂 ml/                    # 머신러닝
│   ├── 📂 models/           # ML 모델
│   └── 📂 optimization/     # 파라미터 최적화
├── 📂 deployment/            # 배포 관련
│   ├── gvs_deploy.py       # GVS 서버 배포
│   └── deployment_config.json
├── 📂 config/               # 설정 파일
├── 📂 data/                 # 데이터 저장
├── 📂 logs/                 # 로그 파일
└── start_system.py          # 시스템 런처
```

## 🔧 설치 및 설정

### 1. 필수 패키지 설치

```bash
pip install flask flask-cors pandas numpy ccxt scikit-learn xgboost
pip install paramiko scp optuna asyncio aiohttp
```

### 2. 환경 변수 설정

```bash
# 바이낸스 API 키 설정 (선택사항)
set BINANCE_API_KEY=your_api_key
set BINANCE_API_SECRET=your_api_secret
```

### 3. 방화벽 설정

- **로컬 테스트**: 9000번 포트 허용
- **GVS 서버**: Google Cloud Platform 방화벽 규칙에서 9000번 포트 허용

## 🚀 시스템 실행

### 방법 1: 간단한 대시보드 실행

```bash
# Windows
start_dashboard.bat

# 또는 직접 실행
python dashboard/app.py
```

### 방법 2: 통합 시스템 실행

```bash
# 전체 시스템 실행
python start_system.py full

# 대시보드만 실행
python start_system.py dashboard

# 실전매매만 실행
python start_system.py trading

# 백테스트만 실행
python start_system.py backtest
```

## 🌐 웹 대시보드 사용법

### 접속 주소

- **로컬**: http://localhost:9000
- **GVS 서버**: http://34.47.77.230:9000

### 페이지 구성

1. **메인 대시보드** (`/`)
   - 시스템 상태 확인
   - 백테스트/실전매매 선택
   - 최근 성과 요약

2. **백테스트 페이지** (`/backtest`)
   - 전략 설정
   - 데이터 기간 선택
   - 결과 분석

3. **실전매매 페이지** (`/live-trading`)
   - 거래 제어
   - 포지션 모니터링
   - 실시간 상태

## 🔥 실전매매 시스템

### 주요 특징

- **24/7 자동 운영**: 시장이 열려있는 동안 지속적 모니터링
- **다중 전략**: 트리플 콤보, 스캘핑, 트렌드 추종
- **리스크 관리**: 동적 손절/익절, 드로다운 관리
- **실시간 알림**: 텔레그램 봇 지원

### 설정 예시

```json
{
  "initial_capital": 10000000,
  "max_position_size": 0.1,
  "stop_loss_ratio": 0.02,
  "take_profit_ratio": 0.05,
  "confidence_threshold": 0.6,
  "max_daily_loss": 0.05
}
```

## 📊 백테스트 시스템

### 지원 전략

1. **트리플 콤보**: ML + 기술적 분석 + 볼륨 분석
2. **스캘핑**: 단기 가격 변동 포착
3. **트렌드 추종**: 중장기 추세 추종

### ML 최적화

- **Optuna 기반**: 베이지안 최적화
- **파라미터 자동 탐색**: 100+ 파라미터 조합
- **백테스트 검증**: 과최적화 방지

## 🏗️ GVS 서버 배포

### 자동 배포

```bash
# Windows
deploy_to_gvs.bat

# 또는 직접 실행
python deployment/gvs_deploy.py deploy
```

### 배포 프로세스

1. **패키지 생성**: 프로젝트 압축
2. **서버 업로드**: SSH/SCP를 통한 파일 전송
3. **의존성 설치**: pip install
4. **서비스 설정**: systemd 서비스 등록
5. **방화벽 설정**: 9000번 포트 허용
6. **서비스 시작**: 자동 시작

### 서버 관리

```bash
# 서버 상태 확인
python deployment/gvs_deploy.py status

# 서비스 재시작
python deployment/gvs_deploy.py restart
```

## 🎯 바이낸스 선물 연동

### 지원 기능

- **USDT.P 심볼**: 모든 USDT 무기한 선물 지원
- **실시간 데이터**: WebSocket 연결
- **거래량 분석**: 상위 거래량 심볼 자동 선별
- **트렌딩 분석**: 시장 상황에 맞는 핫 유니버스

### 심볼 선택 방법

```python
# 거래량 상위 50개
top_symbols = await api.get_top_volume_symbols(50)

# 트렌딩 상위 20개
hot_symbols = await api.get_market_trending_symbols(20)

# 전체 USDT.P 심볼
all_symbols = await api.get_usdt_perpetual_symbols()
```

## 🧠 ML 최적화 시스템

### 최적화 파라미터

- **ML 모델**: 룩백 기간, 예측 호라이즌, 피처 중요도
- **전략 설정**: 신뢰도 임계값, 추세 감지 민감도
- **리스크 관리**: 포지션 크기, 손절/익절 비율
- **기술적 지표**: RSI, MACD, 볼린저 밴드 파라미터

### 최적화 실행

```python
# 파라미터 최적화
optimizer = ParameterOptimizer()
result = await optimizer.optimize_parameters(
    symbols=['BTC/USDT', 'ETH/USDT'],
    strategy_type='triple_combo',
    n_trials=100
)
```

## 📈 성과 분석

### 지표 계산

- **수익률**: 총 수익률, 연간 수익률
- **리스크 지표**: 샤프 비율, 최대 낙폭, VaR
- **거래 통계**: 승률, 평균 손익, 수익 인수

### 시각화

- **수익률 곡선**: 시간대별 성과
- **드로다운 차트**: 위험 구간 분석
- **거래 분포**: 손익 히스토그램

## 🔒 보안 및 위험 관리

### 보안 조치

- **API 키 암호화**: 환경 변수 사용
- **네트워크 보안**: HTTPS, 방화벽 설정
- **접근 제어**: IP 화이트리스트 (선택사항)

### 위험 관리

- **포지션 크기 제한**: 최대 포지션 10%
- **손절/익절**: 동적 스탑로스 적용
- **드로다운 관리**: 최대 15% 제한
- **긴급 정지**: 원클릭 전체 중지

## 📞 지원 및 문의

### 로그 확인

```bash
# 시스템 로그
tail -f logs/system.log

# 거래 로그
tail -f logs/trading.log

# ML 로그
tail -f logs/ml.log
```

### 문제 해결

1. **연결 오류**: 방화벽 설정 확인
2. **API 오류**: 키 유효성 검사
3. **성능 문제**: 시스템 리소스 확인
4. **거래 오류**: 잔고 및 권한 확인

### 업데이트

```bash
# 코드 업데이트
git pull origin main

# 라이브러리 업데이트
pip install -r requirements.txt --upgrade

# 시스템 재시작
python start_system.py full
```

## 🎉 시작하기

1. **환경 설정**: Python 3.8+ 설치
2. **프로젝트 다운로드**: 코드 복사
3. **패키지 설치**: pip install
4. **대시보드 실행**: `start_dashboard.bat`
5. **웹 접속**: http://localhost:9000

---

**📧 AlphaGenesis Team**  
**🌐 Website**: [AlphaGenesis Dashboard](http://34.47.77.230:9000)  
**📅 Last Updated**: 2025-01-11

> ⚠️ **주의사항**: 실전매매는 높은 위험을 수반합니다. 충분한 테스트 후 소액으로 시작하세요.