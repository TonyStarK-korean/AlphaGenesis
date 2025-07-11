// 메인 대시보드 JavaScript

// 전역 변수
let systemStatus = {
    server: true,
    data: true,
    binance: true,
    ml: true
};

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    startStatusUpdates();
});

// 대시보드 초기화
function initializeDashboard() {
    updateSystemStatus();
    updatePerformanceMetrics();
    
    // 서버 연결 확인
    checkServerConnection();
    
    console.log('🚀 AlphaGenesis 대시보드 초기화 완료');
}

// 백테스트 페이지로 이동
function navigateToBacktest() {
    showLoadingMessage('백테스트 시스템으로 이동 중...');
    
    setTimeout(() => {
        window.location.href = '/backtest';
    }, 1000);
}

// 실전매매 페이지로 이동
function navigateToLiveTrading() {
    showLoadingMessage('실전매매 시스템으로 이동 중...');
    
    setTimeout(() => {
        window.location.href = '/live-trading';
    }, 1000);
}

// 시스템 상태 업데이트
function updateSystemStatus() {
    // 데이터 상태
    const dataStatus = document.getElementById('dataStatus');
    if (systemStatus.data) {
        dataStatus.textContent = '실시간 업데이트 중';
        dataStatus.style.color = '#10b981';
    } else {
        dataStatus.textContent = '연결 끊김';
        dataStatus.style.color = '#ef4444';
    }
    
    // 바이낸스 상태
    const binanceStatus = document.getElementById('binanceStatus');
    if (systemStatus.binance) {
        binanceStatus.textContent = '연결됨';
        binanceStatus.style.color = '#10b981';
    } else {
        binanceStatus.textContent = '연결 실패';
        binanceStatus.style.color = '#ef4444';
    }
    
    // ML 모델 상태
    const mlStatus = document.getElementById('mlStatus');
    if (systemStatus.ml) {
        mlStatus.textContent = '최적화 완료';
        mlStatus.style.color = '#10b981';
    } else {
        mlStatus.textContent = '최적화 중';
        mlStatus.style.color = '#f59e0b';
    }
    
    // 전략 상태
    const strategyStatus = document.getElementById('strategyStatus');
    strategyStatus.textContent = '트리플 콤보';
    strategyStatus.style.color = '#10b981';
}

// 성과 지표 업데이트
function updatePerformanceMetrics() {
    // 실제 API에서 데이터를 가져오는 부분 (현재는 더미 데이터)
    const metrics = {
        todayReturn: 2.34,
        weekReturn: 8.76,
        maxDrawdown: -3.21,
        sharpeRatio: 2.41
    };
    
    document.getElementById('todayReturn').textContent = `${metrics.todayReturn > 0 ? '+' : ''}${metrics.todayReturn}%`;
    document.getElementById('weekReturn').textContent = `${metrics.weekReturn > 0 ? '+' : ''}${metrics.weekReturn}%`;
    document.getElementById('maxDrawdown').textContent = `${metrics.maxDrawdown}%`;
    document.getElementById('sharpeRatio').textContent = metrics.sharpeRatio.toFixed(2);
    
    // 색상 업데이트
    const todayElement = document.getElementById('todayReturn');
    const weekElement = document.getElementById('weekReturn');
    
    todayElement.className = `performance-value ${metrics.todayReturn > 0 ? 'positive' : 'negative'}`;
    weekElement.className = `performance-value ${metrics.weekReturn > 0 ? 'positive' : 'negative'}`;
}

// 서버 연결 확인
async function checkServerConnection() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.status === 'ok') {
            systemStatus.server = true;
            updateServerStatus(true);
        } else {
            systemStatus.server = false;
            updateServerStatus(false);
        }
    } catch (error) {
        console.error('서버 연결 확인 실패:', error);
        systemStatus.server = false;
        updateServerStatus(false);
    }
}

// 서버 상태 UI 업데이트
function updateServerStatus(isConnected) {
    const statusIndicator = document.getElementById('serverStatus');
    const statusIcon = statusIndicator.querySelector('i');
    const statusText = statusIndicator.querySelector('span');
    
    if (isConnected) {
        statusIcon.style.color = '#10b981';
        statusText.textContent = '서버 연결됨';
        statusIndicator.style.color = '#10b981';
    } else {
        statusIcon.style.color = '#ef4444';
        statusText.textContent = '서버 연결 실패';
        statusIndicator.style.color = '#ef4444';
    }
}

// 정기적 상태 업데이트
function startStatusUpdates() {
    // 5초마다 상태 확인
    setInterval(() => {
        checkServerConnection();
        updateSystemStatus();
        updatePerformanceMetrics();
    }, 5000);
    
    // 업타임 업데이트
    setInterval(updateUptime, 1000);
}

// 업타임 업데이트
function updateUptime() {
    const uptimeElement = document.getElementById('uptime');
    if (uptimeElement) {
        const startTime = new Date('2025-01-01T00:00:00');
        const now = new Date();
        const diff = now - startTime;
        
        const days = Math.floor(diff / (1000 * 60 * 60 * 24));
        const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        
        uptimeElement.textContent = `${days}일 ${hours}시간 ${minutes}분`;
    }
}

// 시스템 정보 모달
function showSystemInfo() {
    const modal = document.getElementById('systemModal');
    modal.style.display = 'block';
    
    // 시스템 정보 업데이트
    updateSystemInfo();
}

// 시스템 정보 업데이트
function updateSystemInfo() {
    // 메모리 사용량 (더미 데이터)
    const memoryUsage = Math.floor(Math.random() * 30) + 50;
    document.getElementById('memoryUsage').textContent = `${memoryUsage}%`;
    
    // CPU 사용량 (더미 데이터)
    const cpuUsage = Math.floor(Math.random() * 40) + 10;
    document.getElementById('cpuUsage').textContent = `${cpuUsage}%`;
}

// 도움말 표시
function showHelp() {
    alert(`
🚀 AlphaGenesis 트레이딩 시스템

📊 백테스트:
- 과거 데이터로 전략 성능 검증
- ML 최적화 지원
- 모든 USDT.P 심볼 지원

🔥 실전매매:
- 24/7 자동 트레이딩
- 실시간 리스크 관리
- 텔레그램 알림 지원

💡 도움이 필요하면 문의하세요!
    `);
}

// 설정 표시
function showSettings() {
    alert('설정 페이지는 개발 중입니다. 곧 추가될 예정입니다.');
}

// 모달 닫기
function closeModal() {
    const modal = document.getElementById('systemModal');
    modal.style.display = 'none';
}

// 모달 외부 클릭 시 닫기
window.onclick = function(event) {
    const modal = document.getElementById('systemModal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
}

// 로딩 메시지 표시
function showLoadingMessage(message) {
    // 간단한 로딩 표시
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loadingMessage';
    loadingDiv.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        z-index: 3000;
        text-align: center;
        font-size: 1.1rem;
    `;
    
    loadingDiv.innerHTML = `
        <div class="loading"></div>
        <p style="margin-top: 1rem;">${message}</p>
    `;
    
    document.body.appendChild(loadingDiv);
    
    // 3초 후 제거
    setTimeout(() => {
        const loading = document.getElementById('loadingMessage');
        if (loading) {
            loading.remove();
        }
    }, 3000);
}

// 키보드 단축키
document.addEventListener('keydown', function(e) {
    // Ctrl + B: 백테스트
    if (e.ctrlKey && e.key === 'b') {
        e.preventDefault();
        navigateToBacktest();
    }
    
    // Ctrl + L: 실전매매
    if (e.ctrlKey && e.key === 'l') {
        e.preventDefault();
        navigateToLiveTrading();
    }
    
    // Ctrl + I: 시스템 정보
    if (e.ctrlKey && e.key === 'i') {
        e.preventDefault();
        showSystemInfo();
    }
});

// 에러 핸들링
window.addEventListener('error', function(e) {
    console.error('JavaScript 에러:', e);
    
    // 사용자에게 알림
    if (e.error && e.error.message) {
        showErrorMessage('시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
    }
});

// 에러 메시지 표시
function showErrorMessage(message) {
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #ef4444;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        z-index: 3000;
        max-width: 300px;
        animation: slideIn 0.3s ease;
    `;
    
    errorDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle"></i>
        <span style="margin-left: 0.5rem;">${message}</span>
    `;
    
    document.body.appendChild(errorDiv);
    
    // 5초 후 제거
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// 성공 메시지 표시
function showSuccessMessage(message) {
    const successDiv = document.createElement('div');
    successDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #10b981;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        z-index: 3000;
        max-width: 300px;
        animation: slideIn 0.3s ease;
    `;
    
    successDiv.innerHTML = `
        <i class="fas fa-check-circle"></i>
        <span style="margin-left: 0.5rem;">${message}</span>
    `;
    
    document.body.appendChild(successDiv);
    
    // 3초 후 제거
    setTimeout(() => {
        successDiv.remove();
    }, 3000);
}

// 애니메이션 CSS 추가
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);