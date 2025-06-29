# Auto Trading System - 프로젝트 구조

## 📁 폴더 구조

```
Auto_trading_system/
├── 📁 core/                          # 핵심 시스템
│   ├── 📁 trading_engine/            # 거래 엔진
│   │   ├── dynamic_leverage_manager.py
│   │   ├── trading_engine.py
│   │   └── order_manager.py
│   ├── 📁 risk_management/           # 리스크 관리
│   │   ├── risk_calculator.py
│   │   ├── position_sizer.py
│   │   └── stop_loss_manager.py
│   ├── 📁 position_management/       # 포지션 관리
│   │   ├── position_tracker.py
│   │   ├── portfolio_manager.py
│   │   └── balance_manager.py
│   ├── main_trading_system.py        # 메인 거래 시스템
│   ├── ultimate_trading_system.py    # 고급 거래 시스템
│   ├── main.py                       # 기본 실행 파일
│   └── advanced_main.py              # 고급 실행 파일
│
├── 📁 strategies/                    # 거래 전략
│   ├── 📁 phase1/                    # Phase1 (공격 모드)
│   │   ├── aggressive_strategy.py
│   │   ├── volatility_breakout.py
│   │   └── momentum_strategy.py
│   ├── 📁 phase2/                    # Phase2 (방어 모드)
│   │   ├── defensive_strategy.py
│   │   ├── mean_reversion.py
│   │   └── trend_following.py
│   └── 📁 ml_strategies/             # ML 기반 전략
│       ├── ml_prediction_strategy.py
│       ├── ensemble_strategy.py
│       └── adaptive_strategy.py
│
├── 📁 ml/                           # 머신러닝
│   ├── 📁 models/                   # ML 모델
│   │   ├── price_prediction_model.py
│   │   ├── market_regime_model.py
│   │   └── risk_model.py
│   ├── 📁 features/                 # 특성 엔지니어링
│   │   ├── feature_extractor.py
│   │   ├── technical_features.py
│   │   └── market_features.py
│   └── 📁 evaluation/               # 모델 평가
│       ├── model_evaluator.py
│       ├── backtest_evaluator.py
│       └── performance_metrics.py
│
├── 📁 data/                         # 데이터 관리
│   ├── 📁 market_data/              # 시장 데이터
│   │   ├── data_generator.py
│   │   ├── data_downloader.py
│   │   └── data_cleaner.py
│   ├── 📁 backtest_data/            # 백테스트 데이터
│   │   ├── historical_data.py
│   │   ├── sample_data.py
│   │   └── backtest_results.py
│   └── 📁 ml_data/                  # ML 데이터
│       ├── training_data.py
│       ├── validation_data.py
│       └── test_data.py
│
├── 📁 utils/                        # 유틸리티
│   ├── 📁 indicators/               # 기술적 지표
│   │   ├── technical_indicators.py
│   │   ├── custom_indicators.py
│   │   └── indicator_calculator.py
│   ├── 📁 calculators/              # 계산기
│   │   ├── profit_calculator.py
│   │   ├── risk_calculator.py
│   │   └── performance_calculator.py
│   └── 📁 validators/               # 검증기
│       ├── data_validator.py
│       ├── signal_validator.py
│       └── config_validator.py
│
├── 📁 dashboard/                    # 웹 대시보드
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── 📁 notification/                 # 알림 시스템
│   ├── telegram_bot.py
│   ├── email_notifier.py
│   └── alert_manager.py
│
├── 📁 exchange/                     # 거래소 연동
│   ├── binance_client.py
│   ├── upbit_client.py
│   └── exchange_manager.py
│
├── 📁 analysis/                     # 분석 도구
│   ├── market_analysis.py
│   ├── performance_analysis.py
│   └── risk_analysis.py
│
├── 📁 config/                       # 설정 파일
│   ├── trading_config.py
│   ├── backtest_config.py
│   └── system_config.py
│
├── 📁 logs/                         # 로그 파일
├── 📁 docs/                         # 문서
├── requirements.txt                 # 의존성
├── run_dashboard.py                 # 대시보드 실행
├── run_ml_backtest.py               # ML 백테스트 실행
└── README.md                        # 프로젝트 설명
```

## 🏗️ 주요 뼈대별 구성

### 1. Core (핵심 시스템)
- **trading_engine**: 거래 실행 엔진
- **risk_management**: 리스크 관리 시스템
- **position_management**: 포지션 관리 시스템

### 2. Strategies (거래 전략)
- **phase1**: 공격적 거래 전략 (소액 알트코인)
- **phase2**: 방어적 거래 전략 (대형 코인)
- **ml_strategies**: ML 기반 전략

### 3. ML (머신러닝)
- **models**: 예측 모델 (Random Forest, XGBoost, LSTM)
- **features**: 특성 엔지니어링
- **evaluation**: 모델 평가 및 성능 측정

### 4. Data (데이터 관리)
- **market_data**: 실시간/과거 시장 데이터
- **backtest_data**: 백테스트용 데이터
- **ml_data**: ML 모델용 데이터

### 5. Utils (유틸리티)
- **indicators**: 기술적 지표 계산
- **calculators**: 수익률, 리스크 계산
- **validators**: 데이터 및 설정 검증

## 🚀 주요 기능

### ML 모델 (몇 년치 백테스트 가능)
- **Random Forest**: 안정적인 예측
- **XGBoost**: 고성능 부스팅 모델
- **LSTM**: 시계열 예측
- **앙상블**: 다중 모델 결합

### 동적 레버리지 시스템
- **시장 국면별 조정**: 상승장/하락장/횡보장
- **Phase별 조정**: Phase1(공격)/Phase2(방어)
- **리스크 기반 조정**: 낙폭, 연속 손실 등

### Phase별 전략
- **Phase1**: 소액 알트코인 공격 모드 (최대 7배 레버리지)
- **Phase2**: 대형 코인 방어 모드 (최대 5배 레버리지)

## 📊 백테스트 기능

### ML 백테스트
```bash
python run_ml_backtest.py
```

- 3년치 과거 데이터로 백테스트
- 동적 레버리지 적용
- ML 모델 성능 평가
- 상세한 결과 분석

### 웹 대시보드
```bash
python run_dashboard.py
```

- 실시간 거래 현황
- 백테스트 결과 시각화
- 성능 지표 대시보드

## 🔧 설정

### 레버리지 설정
- **Phase1**: 기본 3배, 최대 7배, 최소 1.5배
- **Phase2**: 기본 1.5배, 최대 5배, 최소 1배

### 시장 국면별 조정
- **상승장**: +30% (Phase1), +40% (Phase2)
- **하락장**: -40% (Phase1), -30% (Phase2)
- **고변동성**: -50% (Phase1), -40% (Phase2)
- **저변동성**: +20% (Phase1), +30% (Phase2)

## 📈 성능 지표

- **총 수익률**: 백테스트 기간 동안의 총 수익률
- **최대 낙폭**: 최대 손실 구간
- **승률**: 수익 거래 비율
- **샤프 비율**: 위험 대비 수익률
- **평균 레버리지**: 거래 기간 평균 레버리지

## 🛡️ 리스크 관리

- **동적 레버리지**: 시장 상황에 따른 자동 조정
- **손절매**: 자동 손절매 시스템
- **포지션 사이징**: 리스크 기반 포지션 크기 조정
- **분산 투자**: 다중 코인 분산 투자

이 구조는 상위 0.01%급 자동매매 시스템을 위한 체계적이고 확장 가능한 아키텍처를 제공합니다. 