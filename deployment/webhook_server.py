#!/usr/bin/env python3
"""
GitHub Webhook 서버
GitHub 푸시 이벤트를 실시간으로 받아서 즉시 업데이트
"""

import os
import sys
import json
import hmac
import hashlib
import subprocess
import logging
from datetime import datetime
from flask import Flask, request, jsonify
import threading
import signal

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/alphagenesis_webhook.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GitHubWebhookServer:
    """GitHub Webhook 서버 클래스"""
    
    def __init__(self, config_file="/root/AlphaGenesis/deployment/webhook_config.json"):
        """초기화"""
        self.config_file = config_file
        self.app = Flask(__name__)
        
        # 기본 설정
        self.config = {
            "webhook": {
                "port": 8080,
                "secret": "",  # GitHub Webhook Secret
                "branch": "main"
            },
            "server": {
                "repo_path": "/root/AlphaGenesis",
                "service_name": "alphagenesis",
                "auto_restart": True
            },
            "security": {
                "verify_signature": True,
                "allowed_ips": ["140.82.112.0/20", "192.30.252.0/22"]  # GitHub IP ranges
            }
        }
        
        # 설정 로드
        self.load_config()
        
        # 라우트 설정
        self.setup_routes()
        
        logger.info("GitHub Webhook 서버 초기화 완료")
    
    def load_config(self):
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                logger.info("Webhook 설정 파일 로드 완료")
            else:
                # 기본 설정 파일 생성
                os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                logger.info("기본 Webhook 설정 파일 생성")
                
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
    
    def setup_routes(self):
        """Flask 라우트 설정"""
        
        @self.app.route('/')
        def index():
            """루트 경로"""
            return jsonify({
                "service": "AlphaGenesis GitHub Webhook",
                "status": "running",
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/webhook', methods=['POST'])
        def webhook():
            """GitHub Webhook 엔드포인트"""
            try:
                # 요청 검증
                if not self.verify_request(request):
                    logger.warning("Webhook 요청 검증 실패")
                    return jsonify({"error": "Unauthorized"}), 401
                
                # 페이로드 파싱
                payload = request.get_json()
                
                if not payload:
                    logger.warning("빈 페이로드 수신")
                    return jsonify({"error": "Empty payload"}), 400
                
                # 이벤트 처리
                return self.handle_webhook_event(payload)
                
            except Exception as e:
                logger.error(f"Webhook 처리 실패: {e}")
                return jsonify({"error": "Internal server error"}), 500
        
        @self.app.route('/health')
        def health():
            """헬스 체크"""
            return jsonify({"status": "healthy"})
    
    def verify_request(self, request):
        """요청 검증"""
        try:
            # IP 검증 (선택사항)
            client_ip = request.remote_addr
            if self.config["security"]["allowed_ips"]:
                # 실제 구현에서는 IP 범위 체크 로직 필요
                pass
            
            # 시그니처 검증
            if self.config["security"]["verify_signature"]:
                signature = request.headers.get('X-Hub-Signature-256')
                if not signature:
                    return False
                
                secret = self.config["webhook"]["secret"]
                if not secret:
                    logger.warning("Webhook secret이 설정되지 않았습니다.")
                    return True  # secret이 없으면 검증 스킵
                
                expected_signature = hmac.new(
                    secret.encode(),
                    request.data,
                    hashlib.sha256
                ).hexdigest()
                
                expected_signature = f"sha256={expected_signature}"
                
                if not hmac.compare_digest(signature, expected_signature):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"요청 검증 실패: {e}")
            return False
    
    def handle_webhook_event(self, payload):
        """Webhook 이벤트 처리"""
        try:
            # 이벤트 타입 확인
            event_type = request.headers.get('X-GitHub-Event')
            
            if event_type == 'ping':
                logger.info("GitHub ping 이벤트 수신")
                return jsonify({"message": "pong"})
            
            elif event_type == 'push':
                return self.handle_push_event(payload)
            
            elif event_type == 'release':
                return self.handle_release_event(payload)
            
            else:
                logger.info(f"무시된 이벤트 타입: {event_type}")
                return jsonify({"message": "Event ignored"})
                
        except Exception as e:
            logger.error(f"이벤트 처리 실패: {e}")
            return jsonify({"error": "Event processing failed"}), 500
    
    def handle_push_event(self, payload):
        """Push 이벤트 처리"""
        try:
            # 브랜치 확인
            ref = payload.get('ref', '')
            target_branch = f"refs/heads/{self.config['webhook']['branch']}"
            
            if ref != target_branch:
                logger.info(f"대상 브랜치가 아님: {ref}")
                return jsonify({"message": "Branch ignored"})
            
            # 커밋 정보 추출
            commits = payload.get('commits', [])
            if not commits:
                logger.info("커밋이 없음")
                return jsonify({"message": "No commits"})
            
            latest_commit = commits[-1]
            commit_info = {
                "id": latest_commit.get('id', ''),
                "message": latest_commit.get('message', ''),
                "author": latest_commit.get('author', {}).get('name', ''),
                "timestamp": latest_commit.get('timestamp', '')
            }
            
            logger.info(f"Push 이벤트 수신: {commit_info['id'][:8]} - {commit_info['message']}")
            
            # 백그라운드에서 업데이트 실행
            update_thread = threading.Thread(
                target=self.perform_update,
                args=(commit_info,)
            )
            update_thread.daemon = True
            update_thread.start()
            
            return jsonify({
                "message": "Update triggered",
                "commit": commit_info
            })
            
        except Exception as e:
            logger.error(f"Push 이벤트 처리 실패: {e}")
            return jsonify({"error": "Push event processing failed"}), 500
    
    def handle_release_event(self, payload):
        """Release 이벤트 처리"""
        try:
            action = payload.get('action', '')
            
            if action == 'published':
                release = payload.get('release', {})
                tag_name = release.get('tag_name', '')
                name = release.get('name', '')
                
                logger.info(f"새 릴리스 발행: {tag_name} - {name}")
                
                # 릴리스에 대한 특별한 처리가 필요한 경우 여기에 구현
                
                return jsonify({
                    "message": "Release processed",
                    "tag": tag_name,
                    "name": name
                })
            
            return jsonify({"message": "Release action ignored"})
            
        except Exception as e:
            logger.error(f"Release 이벤트 처리 실패: {e}")
            return jsonify({"error": "Release event processing failed"}), 500
    
    def perform_update(self, commit_info):
        """업데이트 수행 (백그라운드)"""
        try:
            logger.info("자동 업데이트 시작...")
            
            repo_path = self.config["server"]["repo_path"]
            service_name = self.config["server"]["service_name"]
            
            # 1. 백업 생성
            self.create_backup()
            
            # 2. Git pull
            logger.info("Git pull 실행...")
            pull_cmd = f"cd {repo_path} && git pull origin {self.config['webhook']['branch']}"
            result = subprocess.run(pull_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Git pull 실패: {result.stderr}")
                return
            
            logger.info("Git pull 완료")
            
            # 3. 의존성 설치
            self.install_dependencies()
            
            # 4. 서비스 재시작
            if self.config["server"]["auto_restart"]:
                logger.info("서비스 재시작...")
                restart_cmd = f"systemctl restart {service_name}"
                result = subprocess.run(restart_cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("서비스 재시작 완료")
                else:
                    logger.error(f"서비스 재시작 실패: {result.stderr}")
                    return
            
            # 5. 상태 확인
            self.verify_service_status()
            
            logger.info(f"자동 업데이트 완료 - 커밋: {commit_info['id'][:8]}")
            
        except Exception as e:
            logger.error(f"업데이트 수행 실패: {e}")
    
    def create_backup(self):
        """백업 생성"""
        try:
            repo_path = self.config["server"]["repo_path"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"alphagenesis_webhook_backup_{timestamp}.tar.gz"
            backup_dir = f"{repo_path}_backups"
            
            os.makedirs(backup_dir, exist_ok=True)
            
            backup_path = os.path.join(backup_dir, backup_name)
            backup_cmd = f"tar -czf {backup_path} -C {os.path.dirname(repo_path)} {os.path.basename(repo_path)}"
            
            result = subprocess.run(backup_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"백업 생성 완료: {backup_path}")
            else:
                logger.warning(f"백업 생성 실패: {result.stderr}")
                
        except Exception as e:
            logger.error(f"백업 생성 실패: {e}")
    
    def install_dependencies(self):
        """의존성 설치"""
        try:
            repo_path = self.config["server"]["repo_path"]
            requirements_file = os.path.join(repo_path, "requirements.txt")
            
            if os.path.exists(requirements_file):
                install_cmd = f"cd {repo_path} && pip3 install -r requirements.txt --upgrade"
                result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("의존성 업데이트 완료")
                else:
                    logger.warning(f"의존성 업데이트 경고: {result.stderr}")
                    
        except Exception as e:
            logger.error(f"의존성 설치 실패: {e}")
    
    def verify_service_status(self):
        """서비스 상태 확인"""
        try:
            service_name = self.config["server"]["service_name"]
            
            # 서비스 상태 확인
            status_cmd = f"systemctl is-active {service_name}"
            result = subprocess.run(status_cmd, shell=True, capture_output=True, text=True)
            
            status = result.stdout.strip()
            if status == "active":
                logger.info("서비스가 정상적으로 실행 중")
            else:
                logger.error(f"서비스 상태 이상: {status}")
                
        except Exception as e:
            logger.error(f"서비스 상태 확인 실패: {e}")
    
    def run(self):
        """웹훅 서버 실행"""
        try:
            port = self.config["webhook"]["port"]
            logger.info(f"GitHub Webhook 서버 시작 (포트: {port})")
            
            self.app.run(
                host='0.0.0.0',
                port=port,
                debug=False,
                use_reloader=False
            )
            
        except Exception as e:
            logger.error(f"웹훅 서버 실행 실패: {e}")
            raise

def create_webhook_service():
    """systemd 서비스 파일 생성"""
    service_content = """[Unit]
Description=AlphaGenesis GitHub Webhook Server
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 /root/AlphaGenesis/deployment/webhook_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    service_path = "/etc/systemd/system/alphagenesis-webhook.service"
    
    try:
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "enable", "alphagenesis-webhook"], check=True)
        
        print(f"✅ Webhook 서비스 생성 완료: {service_path}")
        print("서비스 시작: sudo systemctl start alphagenesis-webhook")
        print("서비스 상태: sudo systemctl status alphagenesis-webhook")
        
        # 방화벽 설정
        subprocess.run(["ufw", "allow", "8080/tcp"], check=True)
        print("✅ 방화벽 포트 8080 허용 완료")
        
    except Exception as e:
        print(f"❌ Webhook 서비스 생성 실패: {e}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AlphaGenesis GitHub Webhook 서버")
    parser.add_argument("action", choices=["start", "install-service"], 
                       nargs="?", default="start",
                       help="수행할 작업")
    
    args = parser.parse_args()
    
    if args.action == "install-service":
        create_webhook_service()
    else:
        # 시그널 핸들러 설정
        def signal_handler(signum, frame):
            logger.info("종료 시그널 수신")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 웹훅 서버 시작
        webhook_server = GitHubWebhookServer()
        webhook_server.run()

if __name__ == "__main__":
    main()