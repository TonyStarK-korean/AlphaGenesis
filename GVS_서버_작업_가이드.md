# 🖥️ GVS 서버 백테스트 작업 가이드

## 📋 **작업 순서**

### **1. SSH 접속**
```bash
# GVS 서버 접속
ssh [사용자명]@34.47.77.230

# 또는 포트 지정이 필요한 경우
ssh -p [포트번호] [사용자명]@34.47.77.230
```

### **2. 작업 디렉토리 이동**
```bash
# AlphaGenesis 디렉토리로 이동
cd /path/to/AlphaGenesis

# 현재 디렉토리 확인
pwd
ls -la
```

### **3. Python 환경 확인 및 패키지 설치**
```bash
# Python 버전 확인
python3 --version

# 필요한 패키지 설치
pip3 install requests flask flask-cors pandas numpy scikit-learn xgboost lightgbm optuna pytz tqdm

# 또는 requirements.txt가 있다면
pip3 install -r requirements.txt
```

### **4. 백테스트 실행**

#### **방법 1: 서버용 통합 실행기 사용 (권장)**
```bash
# 대화형 실행기 실행
python3 run_server_backtest.py

# 선택 메뉴:
# 1. ML 백테스트
# 2. 병렬 백테스트  
# 3. 연결 테스트만
```

#### **방법 2: 개별 스크립트 직접 실행**
```bash
# ML 백테스트 실행
python3 run_ml_backtest.py

# 병렬 백테스트 실행
python3 run_parallel_backtest.py
```

### **5. 실시간 모니터링**

#### **대시보드 접속**
- **로컬 접속**: http://localhost:5000 (서버에서 브라우저 사용 시)
- **원격 접속**: http://34.47.77.230:5000 (외부에서 접속)

#### **터미널에서 실시간 로그 확인**
```bash
# 백테스트 실행 중 다른 터미널에서 로그 확인
tail -f logs/ml_backtest.log

# 또는 실시간 프로세스 확인
ps aux | grep python
```

### **6. 백그라운드 실행**

#### **nohup 사용**
```bash
# 백그라운드에서 실행 (SSH 연결이 끊어져도 계속 실행)
nohup python3 run_ml_backtest.py > backtest_output.log 2>&1 &

# 실행 중인 프로세스 확인
jobs
ps aux | grep python

# 프로세스 종료
kill [PID]
```

#### **screen 사용**
```bash
# screen 세션 시작
screen -S backtest

# 백테스트 실행
python3 run_server_backtest.py

# Ctrl+A, D로 세션에서 빠져나옴 (백그라운드 실행)

# 다시 세션에 접속
screen -r backtest

# 세션 목록 확인
screen -ls
```

## 🔧 **문제 해결**

### **포트 충돌 문제**
```bash
# 포트 5000 사용 중인 프로세스 확인
netstat -tulpn | grep :5000
lsof -i :5000

# 프로세스 종료
kill -9 [PID]
```

### **방화벽 설정**
```bash
# 포트 5000 열기 (Ubuntu/CentOS)
sudo ufw allow 5000
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload
```

### **Python 경로 문제**
```bash
# Python 경로 확인
which python3
which pip3

# 가상환경 사용하는 경우
source venv/bin/activate
```

## 📊 **실행 결과 확인**

### **1. 터미널 출력**
- 백테스트 진행 상황
- 매매 현황
- ML 예측값
- 오류 메시지

### **2. 대시보드 (http://34.47.77.230:5000)**
- 실시간 자산 변화 그래프
- 현재 포지션 상태
- 수익률 현황
- 실시간 로그

### **3. 로그 파일**
```bash
# 로그 파일 확인
cat logs/ml_backtest.log
tail -n 100 logs/ml_backtest.log
```

## ⚡ **성능 최적화 팁**

### **1. 멀티프로세싱 활용**
```bash
# CPU 코어 수 확인
nproc

# 병렬 백테스트로 여러 종목 동시 처리
python3 run_parallel_backtest.py
```

### **2. 메모리 사용량 모니터링**
```bash
# 메모리 사용량 확인
free -h
htop

# Python 프로세스 메모리 사용량
ps aux | grep python | awk '{print $4, $11}'
```

### **3. 디스크 용량 확인**
```bash
# 디스크 사용량 확인
df -h

# 로그 파일 용량 확인
du -sh logs/
```

## 🚨 **주의사항**

1. **SSH 연결 유지**: 장시간 실행 시 screen 또는 nohup 사용
2. **리소스 모니터링**: CPU, 메모리 사용량 주기적 확인
3. **백업**: 중요한 결과는 별도 저장
4. **로그 관리**: 로그 파일이 너무 커지지 않도록 주기적 정리

## 📞 **지원**

문제 발생 시:
1. 터미널 출력 메시지 확인
2. 로그 파일 확인
3. 대시보드 연결 상태 확인
4. 필요시 프로세스 재시작 