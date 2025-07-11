#!/usr/bin/env python3
"""
서버 자동 업데이트 시스템
GitHub 변경사항을 자동으로 감지하고 서버에 반영
"""

import os
import sys
import subprocess
import time
import json
import logging
import signal
import requests
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import threading

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/alphagenesis_auto_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ServerAutoUpdater:
    """서버 자동 업데이트 클래스"""
    
    def __init__(self, config_file="/root/AlphaGenesis/deployment/server_config.json"):
        """초기화"""
        self.config_file = config_file
        self.running = True
        
        # 기본 설정
        self.config = {
            "github": {
                "repository": "TonyStarK-korean/AlphaGenesis",
                "branch": "main",
                "api_url": "https://api.github.com/repos/TonyStarK-korean/AlphaGenesis",
                "check_interval": 300  # 5분마다 체크
            },
            "server": {
                "repo_path": "/root/AlphaGenesis",
                "service_name": "alphagenesis",
                "backup_count": 5,
                "auto_restart": True
            },
            "notification": {
                "enabled": True,
                "webhook_url": "",  # Slack/Discord 웹훅 URL
                "telegram_bot_token": "",
                "telegram_chat_id": ""
            }
        }
        
        # 설정 로드
        self.load_config()
        
        # 상태 변수
        self.last_commit_sha = None
        self.last_check_time = None
        
        logger.info("서버 자동 업데이트 시스템 초기화 완료")
    
    def load_config(self):
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                logger.info("설정 파일 로드 완료")
            else:
                # 기본 설정 파일 생성
                os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                logger.info("기본 설정 파일 생성 완료")
                
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
    
    def get_latest_commit_info(self):
        """GitHub에서 최신 커밋 정보 조회"""
        try:
            api_url = self.config["github"]["api_url"]
            branch = self.config["github"]["branch"]
            
            # GitHub API 호출
            response = requests.get(f"{api_url}/commits/{branch}", timeout=30)
            response.raise_for_status()
            
            commit_data = response.json()
            
            return {
                "sha": commit_data["sha"],
                "message": commit_data["commit"]["message"],
                "author": commit_data["commit"]["author"]["name"],
                "date": commit_data["commit"]["author"]["date"],
                "url": commit_data["html_url"]
            }
            
        except Exception as e:
            logger.error(f"GitHub API 호출 실패: {e}")
            return None
    
    def check_for_updates(self):
        """업데이트 확인"""
        try:
            logger.info("업데이트 확인 중...")
            
            # 최신 커밋 정보 조회
            latest_commit = self.get_latest_commit_info()
            
            if not latest_commit:
                logger.warning("커밋 정보를 가져올 수 없습니다.")
                return False
            
            # 첫 실행인 경우 현재 커밋을 기준으로 설정
            if self.last_commit_sha is None:
                self.last_commit_sha = latest_commit["sha"]
                logger.info(f"첫 실행 - 기준 커밋: {self.last_commit_sha[:8]}")
                return False
            
            # 새로운 커밋이 있는지 확인
            if latest_commit["sha"] != self.last_commit_sha:
                logger.info(f"새로운 업데이트 발견!")
                logger.info(f"이전 커밋: {self.last_commit_sha[:8]}")
                logger.info(f"새 커밋: {latest_commit['sha'][:8]}")
                logger.info(f"커밋 메시지: {latest_commit['message']}")
                logger.info(f"작성자: {latest_commit['author']}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"업데이트 확인 실패: {e}")
            return False
    
    def backup_current_version(self):
        """현재 버전 백업"""
        try:
            logger.info("현재 버전 백업 중...")
            
            repo_path = self.config["server"]["repo_path"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"alphagenesis_auto_backup_{timestamp}.tar.gz"
            backup_dir = f"{repo_path}_backups"
            
            # 백업 디렉토리 생성
            os.makedirs(backup_dir, exist_ok=True)
            
            # 백업 실행
            backup_path = os.path.join(backup_dir, backup_name)
            backup_cmd = f"tar -czf {backup_path} -C {os.path.dirname(repo_path)} {os.path.basename(repo_path)}"
            
            result = subprocess.run(backup_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"백업 완료: {backup_path}")
                
                # 오래된 백업 정리
                self.cleanup_old_backups(backup_dir)
                
                return backup_path
            else:
                logger.error(f"백업 실패: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"백업 실패: {e}")
            return None
    
    def cleanup_old_backups(self, backup_dir):
        """오래된 백업 파일 정리"""
        try:
            backup_count = self.config["server"]["backup_count"]
            
            # 백업 파일 목록 조회 (생성 시간 순)
            backup_files = []
            for file in os.listdir(backup_dir):
                if file.startswith("alphagenesis_auto_backup_"):
                    file_path = os.path.join(backup_dir, file)
                    backup_files.append((file_path, os.path.getctime(file_path)))
            
            # 생성 시간 기준 정렬 (오래된 것부터)
            backup_files.sort(key=lambda x: x[1])
            
            # 설정된 개수를 초과하는 파일 삭제
            if len(backup_files) > backup_count:
                files_to_delete = backup_files[:-backup_count]
                for file_path, _ in files_to_delete:
                    os.remove(file_path)
                    logger.info(f"오래된 백업 삭제: {file_path}")
                    
        except Exception as e:
            logger.error(f"백업 정리 실패: {e}")
    
    def update_from_github(self):
        """GitHub에서 업데이트"""
        try:
            logger.info("GitHub에서 업데이트 중...")
            
            repo_path = self.config["server"]["repo_path"]
            branch = self.config["github"]["branch"]
            
            # Git pull 실행
            pull_cmd = f"cd {repo_path} && git pull origin {branch}"
            result = subprocess.run(pull_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Git pull 완료")
                logger.info(f"출력: {result.stdout}")
                
                # 최신 커밋 SHA 업데이트
                latest_commit = self.get_latest_commit_info()
                if latest_commit:
                    self.last_commit_sha = latest_commit["sha"]
                
                return True
            else:
                logger.error(f"Git pull 실패: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"GitHub 업데이트 실패: {e}")
            return False
    
    def install_dependencies(self):
        """의존성 설치"""
        try:
            logger.info("의존성 업데이트 중...")
            
            repo_path = self.config["server"]["repo_path"]
            requirements_file = os.path.join(repo_path, "requirements.txt")
            
            if os.path.exists(requirements_file):
                install_cmd = f"cd {repo_path} && pip install -r requirements.txt --upgrade"
                result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("의존성 업데이트 완료")
                else:
                    logger.warning(f"의존성 업데이트 경고: {result.stderr}")
            
            return True
            
        except Exception as e:
            logger.error(f"의존성 설치 실패: {e}")
            return False
    
    def restart_service(self):
        """서비스 재시작"""
        try:
            if not self.config["server"]["auto_restart"]:
                logger.info("자동 재시작이 비활성화되어 있습니다.")
                return True
            
            logger.info("서비스 재시작 중...")
            
            service_name = self.config["server"]["service_name"]
            
            # 서비스 재시작
            restart_cmd = f"systemctl restart {service_name}"
            result = subprocess.run(restart_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("서비스 재시작 완료")
                
                # 서비스 상태 확인
                time.sleep(5)
                status_cmd = f"systemctl is-active {service_name}"
                status_result = subprocess.run(status_cmd, shell=True, capture_output=True, text=True)
                
                if status_result.stdout.strip() == "active":
                    logger.info("서비스가 정상적으로 실행 중입니다.")
                    return True
                else:
                    logger.error(f"서비스 상태: {status_result.stdout.strip()}")
                    return False
            else:
                logger.error(f"서비스 재시작 실패: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"서비스 재시작 실패: {e}")
            return False
    
    def send_notification(self, message, is_error=False):
        """알림 전송"""
        try:
            if not self.config["notification"]["enabled"]:
                return
            
            # 텔레그램 알림
            if self.config["notification"]["telegram_bot_token"]:
                self.send_telegram_notification(message)
            
            # 웹훅 알림 (Slack/Discord)
            if self.config["notification"]["webhook_url"]:
                self.send_webhook_notification(message, is_error)
                
        except Exception as e:
            logger.error(f"알림 전송 실패: {e}")
    
    def send_telegram_notification(self, message):
        """텔레그램 알림 전송"""
        try:
            bot_token = self.config["notification"]["telegram_bot_token"]
            chat_id = self.config["notification"]["telegram_chat_id"]
            
            if not bot_token or not chat_id:
                return
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            
            logger.info("텔레그램 알림 전송 완료")
            
        except Exception as e:
            logger.error(f"텔레그램 알림 전송 실패: {e}")
    
    def send_webhook_notification(self, message, is_error=False):
        """웹훅 알림 전송"""
        try:
            webhook_url = self.config["notification"]["webhook_url"]
            
            if not webhook_url:
                return
            
            # Slack/Discord 형식
            color = "danger" if is_error else "good"
            data = {
                "text": message,
                "color": color,
                "username": "AlphaGenesis Auto-Update"
            }
            
            response = requests.post(webhook_url, json=data, timeout=10)
            response.raise_for_status()
            
            logger.info("웹훅 알림 전송 완료")
            
        except Exception as e:
            logger.error(f"웹훅 알림 전송 실패: {e}")
    
    def perform_update(self):
        """업데이트 수행"""
        try:
            logger.info("자동 업데이트 시작...")
            
            # 1. 백업 생성
            backup_path = self.backup_current_version()
            if not backup_path:
                logger.error("백업 실패 - 업데이트 중단")
                return False
            
            # 2. GitHub에서 업데이트
            if not self.update_from_github():
                logger.error("GitHub 업데이트 실패")
                self.send_notification("❌ GitHub 업데이트 실패", is_error=True)
                return False
            
            # 3. 의존성 설치
            if not self.install_dependencies():
                logger.warning("의존성 설치 실패 (계속 진행)")
            
            # 4. 서비스 재시작
            if not self.restart_service():
                logger.error("서비스 재시작 실패")
                self.send_notification("❌ 서비스 재시작 실패", is_error=True)
                return False
            
            # 5. 성공 알림
            latest_commit = self.get_latest_commit_info()
            if latest_commit:
                message = f"""🚀 AlphaGenesis 자동 업데이트 완료
                
커밋: {latest_commit['sha'][:8]}
메시지: {latest_commit['message']}
작성자: {latest_commit['author']}
시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🌐 대시보드: http://34.47.77.230:9000"""
                
                self.send_notification(message)
            
            logger.info("자동 업데이트 완료!")
            return True
            
        except Exception as e:
            logger.error(f"업데이트 수행 실패: {e}")
            self.send_notification(f"❌ 업데이트 실패: {e}", is_error=True)
            return False
    
    def run_update_loop(self):
        """업데이트 루프 실행"""
        logger.info("자동 업데이트 루프 시작...")
        
        while self.running:
            try:
                # 업데이트 확인
                if self.check_for_updates():
                    # 업데이트 수행
                    self.perform_update()
                else:
                    logger.debug("업데이트 없음")
                
                # 대기
                check_interval = self.config["github"]["check_interval"]
                logger.debug(f"{check_interval}초 대기...")
                
                for _ in range(check_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("사용자 중단 요청")
                break
            except Exception as e:
                logger.error(f"업데이트 루프 오류: {e}")
                time.sleep(60)  # 1분 대기 후 재시도
        
        logger.info("자동 업데이트 루프 종료")
    
    def setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            logger.info(f"시그널 {signum} 수신 - 종료 중...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self):
        """자동 업데이트 시작"""
        try:
            logger.info("서버 자동 업데이트 시스템 시작")
            
            # 시그널 핸들러 설정
            self.setup_signal_handlers()
            
            # 초기 커밋 정보 조회
            latest_commit = self.get_latest_commit_info()
            if latest_commit:
                self.last_commit_sha = latest_commit["sha"]
                logger.info(f"초기 커밋: {self.last_commit_sha[:8]}")
            
            # 업데이트 루프 실행
            self.run_update_loop()
            
        except Exception as e:
            logger.error(f"자동 업데이트 시스템 시작 실패: {e}")
            raise

def create_systemd_service():
    """systemd 서비스 파일 생성"""
    service_content = """[Unit]
Description=AlphaGenesis Auto Update Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 /root/AlphaGenesis/deployment/server_auto_update.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    service_path = "/etc/systemd/system/alphagenesis-auto-update.service"
    
    try:
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        # systemd 리로드 및 서비스 활성화
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "enable", "alphagenesis-auto-update"], check=True)
        
        print(f"✅ systemd 서비스 생성 완료: {service_path}")
        print("서비스 시작: sudo systemctl start alphagenesis-auto-update")
        print("서비스 상태: sudo systemctl status alphagenesis-auto-update")
        
    except Exception as e:
        print(f"❌ systemd 서비스 생성 실패: {e}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AlphaGenesis 서버 자동 업데이트")
    parser.add_argument("action", choices=["start", "install-service", "check"], 
                       nargs="?", default="start",
                       help="수행할 작업")
    
    args = parser.parse_args()
    
    if args.action == "install-service":
        create_systemd_service()
    elif args.action == "check":
        updater = ServerAutoUpdater()
        if updater.check_for_updates():
            print("✅ 새로운 업데이트가 있습니다.")
        else:
            print("ℹ️ 업데이트가 없습니다.")
    else:
        # 자동 업데이트 시작
        updater = ServerAutoUpdater()
        updater.start()

if __name__ == "__main__":
    main()