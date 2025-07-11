#!/usr/bin/env python3
"""
GitHub 기반 웹서버 자동 배포 시스템
로컬 → GitHub → GVS 서버 자동 동기화
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
import paramiko
from scp import SCPClient
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitHubDeployment:
    """GitHub 기반 배포 시스템"""
    
    def __init__(self):
        """초기화"""
        self.project_root = Path(__file__).parent
        self.config_file = self.project_root / "deployment" / "git_config.json"
        
        # 기본 설정
        self.config = {
            "github": {
                "repository": "https://github.com/TonyStarK-korean/AlphaGenesis.git",
                "branch": "main",
                "username": "",  # GitHub 사용자명
                "token": ""      # GitHub Personal Access Token
            },
            "server": {
                "host": "34.47.77.230",
                "port": 22,
                "user": "root",
                "password": "",
                "key_file": ""
            },
            "paths": {
                "local_repo": str(self.project_root),
                "server_repo": "/root/AlphaGenesis",
                "backup_dir": "/root/backups"
            },
            "service": {
                "name": "alphagenesis",
                "auto_restart": True,
                "keep_running": True
            }
        }
        
        # 설정 로드 또는 생성
        self.load_or_create_config()
        
        logger.info("GitHub 배포 시스템 초기화 완료")
    
    def load_or_create_config(self):
        """설정 파일 로드 또는 생성"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                logger.info("기존 설정 파일 로드 완료")
            else:
                # 디렉토리 생성
                self.config_file.parent.mkdir(exist_ok=True)
                
                # 기본 설정 파일 생성
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                
                logger.info(f"기본 설정 파일 생성: {self.config_file}")
                print(f"⚠️  설정 파일을 수정해주세요: {self.config_file}")
                
        except Exception as e:
            logger.error(f"설정 파일 처리 실패: {e}")
    
    def setup_git_repository(self):
        """Git 저장소 초기 설정"""
        try:
            logger.info("Git 저장소 설정 중...")
            
            os.chdir(self.project_root)
            
            # Git 초기화 (이미 있다면 스킵)
            if not (self.project_root / ".git").exists():
                subprocess.run(["git", "init"], check=True)
                logger.info("Git 저장소 초기화 완료")
            
            # 원격 저장소 설정
            try:
                subprocess.run(["git", "remote", "remove", "origin"], check=False)
            except:
                pass
            
            repo_url = self.config["github"]["repository"]
            subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
            
            # 기본 브랜치 설정
            subprocess.run(["git", "branch", "-M", "main"], check=False)
            
            logger.info("Git 원격 저장소 설정 완료")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git 설정 실패: {e}")
            raise
        except Exception as e:
            logger.error(f"Git 저장소 설정 실패: {e}")
            raise
    
    def commit_and_push(self, commit_message: str = None):
        """변경사항 커밋 및 푸시"""
        try:
            logger.info("Git 커밋 및 푸시 중...")
            
            os.chdir(self.project_root)
            
            # 변경사항 확인
            result = subprocess.run(["git", "status", "--porcelain"], 
                                  capture_output=True, text=True)
            
            if not result.stdout.strip():
                logger.info("변경사항이 없습니다.")
                return False
            
            # .gitignore 생성/업데이트
            self.create_gitignore()
            
            # 모든 변경사항 추가
            subprocess.run(["git", "add", "."], check=True)
            
            # 커밋 메시지 생성
            if not commit_message:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                commit_message = f"AlphaGenesis 업데이트 - {timestamp}"
            
            # 커밋
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            
            # 푸시
            branch = self.config["github"]["branch"]
            subprocess.run(["git", "push", "-u", "origin", branch], check=True)
            
            logger.info("Git 푸시 완료")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git 커밋/푸시 실패: {e}")
            raise
        except Exception as e:
            logger.error(f"Git 작업 실패: {e}")
            raise
    
    def create_gitignore(self):
        """gitignore 파일 생성"""
        try:
            gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/*.log
*.log

# Data files
data/market_data/*.csv
data/historical_ohlcv/*.csv

# Results
results/
temp/
deployment/temp/

# Config (보안을 위해 실제 설정은 제외)
deployment/deployment_config.json
deployment/git_config.json

# OS
.DS_Store
Thumbs.db

# Backup files
*.bak
*.backup

# Jupyter Notebook
.ipynb_checkpoints/

# pytest
.pytest_cache/

# Coverage
htmlcov/
.coverage
.coverage.*

# mypy
.mypy_cache/

# Archives
*.zip
*.tar.gz
*.7z
"""
            
            gitignore_path = self.project_root / ".gitignore"
            with open(gitignore_path, 'w', encoding='utf-8') as f:
                f.write(gitignore_content)
            
            logger.info("gitignore 파일 생성/업데이트 완료")
            
        except Exception as e:
            logger.error(f"gitignore 생성 실패: {e}")
    
    def deploy_to_server(self):
        """서버에 배포"""
        try:
            logger.info("서버 배포 시작...")
            
            # SSH 연결
            ssh = self.create_ssh_connection()
            
            try:
                # 1. 기존 서비스 중지
                self.stop_service(ssh)
                
                # 2. 백업 생성
                self.backup_current_version(ssh)
                
                # 3. GitHub에서 최신 코드 다운로드
                self.pull_from_github(ssh)
                
                # 4. 의존성 설치
                self.install_dependencies(ssh)
                
                # 5. 서비스 설정 및 시작
                self.setup_and_start_service(ssh)
                
                # 6. 배포 상태 확인
                self.verify_deployment(ssh)
                
                logger.info("서버 배포 완료!")
                
            finally:
                ssh.close()
                
        except Exception as e:
            logger.error(f"서버 배포 실패: {e}")
            raise
    
    def create_ssh_connection(self):
        """SSH 연결 생성"""
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            server_config = self.config["server"]
            
            if server_config["key_file"] and os.path.exists(server_config["key_file"]):
                ssh.connect(
                    hostname=server_config["host"],
                    port=server_config["port"],
                    username=server_config["user"],
                    key_filename=server_config["key_file"]
                )
                logger.info("SSH 키 파일 인증 성공")
            elif server_config["password"]:
                ssh.connect(
                    hostname=server_config["host"],
                    port=server_config["port"],
                    username=server_config["user"],
                    password=server_config["password"]
                )
                logger.info("SSH 패스워드 인증 성공")
            else:
                raise Exception("SSH 인증 정보가 없습니다.")
            
            return ssh
            
        except Exception as e:
            logger.error(f"SSH 연결 실패: {e}")
            raise
    
    def stop_service(self, ssh):
        """서비스 중지"""
        try:
            logger.info("기존 서비스 중지 중...")
            
            service_name = self.config["service"]["name"]
            
            # 서비스 중지
            stdin, stdout, stderr = ssh.exec_command(f"systemctl stop {service_name}")
            stdout.read()  # 명령 완료 대기
            
            # 프로세스 강제 종료 (혹시 모를 경우)
            ssh.exec_command("pkill -f 'python.*dashboard/app.py'")
            ssh.exec_command("pkill -f 'python.*start_system.py'")
            
            logger.info("서비스 중지 완료")
            
        except Exception as e:
            logger.warning(f"서비스 중지 중 오류 (무시): {e}")
    
    def backup_current_version(self, ssh):
        """현재 버전 백업"""
        try:
            logger.info("현재 버전 백업 중...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"alphagenesis_backup_{timestamp}.tar.gz"
            
            server_repo = self.config["paths"]["server_repo"]
            backup_dir = self.config["paths"]["backup_dir"]
            backup_path = f"{backup_dir}/{backup_name}"
            
            # 백업 디렉토리 생성
            ssh.exec_command(f"mkdir -p {backup_dir}")
            
            # 현재 버전 백업
            if self.directory_exists(ssh, server_repo):
                backup_cmd = f"cd {os.path.dirname(server_repo)} && tar -czf {backup_path} {os.path.basename(server_repo)}"
                stdin, stdout, stderr = ssh.exec_command(backup_cmd)
                
                # 명령 완료 대기
                exit_status = stdout.channel.recv_exit_status()
                if exit_status == 0:
                    logger.info(f"백업 완료: {backup_path}")
                else:
                    logger.warning(f"백업 실패: {stderr.read().decode()}")
            
        except Exception as e:
            logger.warning(f"백업 중 오류 (무시): {e}")
    
    def pull_from_github(self, ssh):
        """GitHub에서 최신 코드 다운로드"""
        try:
            logger.info("GitHub에서 최신 코드 다운로드 중...")
            
            server_repo = self.config["paths"]["server_repo"]
            github_repo = self.config["github"]["repository"]
            branch = self.config["github"]["branch"]
            
            # 기존 디렉토리 삭제
            ssh.exec_command(f"rm -rf {server_repo}")
            
            # 새로 클론
            clone_cmd = f"cd {os.path.dirname(server_repo)} && git clone -b {branch} {github_repo} {os.path.basename(server_repo)}"
            stdin, stdout, stderr = ssh.exec_command(clone_cmd)
            
            # 명령 완료 대기
            exit_status = stdout.channel.recv_exit_status()
            if exit_status == 0:
                logger.info("GitHub 코드 다운로드 완료")
            else:
                error_output = stderr.read().decode()
                logger.error(f"GitHub 클론 실패: {error_output}")
                raise Exception(f"Git clone 실패: {error_output}")
            
        except Exception as e:
            logger.error(f"GitHub 코드 다운로드 실패: {e}")
            raise
    
    def install_dependencies(self, ssh):
        """의존성 설치"""
        try:
            logger.info("의존성 설치 중...")
            
            server_repo = self.config["paths"]["server_repo"]
            
            # pip 업그레이드
            ssh.exec_command("pip install --upgrade pip")
            
            # requirements.txt 확인 및 설치
            requirements_path = f"{server_repo}/requirements.txt"
            
            stdin, stdout, stderr = ssh.exec_command(f"ls -la {requirements_path}")
            if stdout.channel.recv_exit_status() == 0:
                # requirements.txt가 있는 경우
                install_cmd = f"cd {server_repo} && pip install -r requirements.txt"
                stdin, stdout, stderr = ssh.exec_command(install_cmd)
                
                exit_status = stdout.channel.recv_exit_status()
                if exit_status == 0:
                    logger.info("requirements.txt 기반 의존성 설치 완료")
                else:
                    logger.warning(f"의존성 설치 경고: {stderr.read().decode()}")
            else:
                # 개별 패키지 설치
                packages = [
                    "flask", "flask-cors", "pandas", "numpy", "ccxt",
                    "scikit-learn", "xgboost", "paramiko", "scp",
                    "optuna", "asyncio", "aiohttp", "python-dotenv"
                ]
                
                for package in packages:
                    install_cmd = f"pip install {package}"
                    ssh.exec_command(install_cmd)
                    time.sleep(1)  # API 제한 방지
                
                logger.info("개별 패키지 설치 완료")
            
        except Exception as e:
            logger.error(f"의존성 설치 실패: {e}")
            raise
    
    def setup_and_start_service(self, ssh):
        """서비스 설정 및 시작"""
        try:
            logger.info("서비스 설정 및 시작 중...")
            
            server_repo = self.config["paths"]["server_repo"]
            service_name = self.config["service"]["name"]
            
            # systemd 서비스 파일 생성
            service_content = f"""[Unit]
Description=AlphaGenesis Trading Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={server_repo}
Environment=PYTHONPATH={server_repo}
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 {server_repo}/dashboard/app.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
            
            # 서비스 파일 생성
            service_path = f"/etc/systemd/system/{service_name}.service"
            temp_service = f"/tmp/{service_name}.service"
            
            # 임시 파일로 서비스 파일 생성
            stdin, stdout, stderr = ssh.exec_command(f"cat > {temp_service} << 'EOF'\n{service_content}EOF")
            stdout.channel.recv_exit_status()
            
            # 서비스 파일 이동
            ssh.exec_command(f"mv {temp_service} {service_path}")
            
            # systemd 리로드
            ssh.exec_command("systemctl daemon-reload")
            
            # 서비스 활성화
            ssh.exec_command(f"systemctl enable {service_name}")
            
            # 서비스 시작
            stdin, stdout, stderr = ssh.exec_command(f"systemctl start {service_name}")
            exit_status = stdout.channel.recv_exit_status()
            
            if exit_status == 0:
                logger.info("서비스 시작 완료")
            else:
                logger.error(f"서비스 시작 실패: {stderr.read().decode()}")
            
            # 방화벽 설정
            ssh.exec_command("ufw allow 9000/tcp")
            
        except Exception as e:
            logger.error(f"서비스 설정 실패: {e}")
            raise
    
    def verify_deployment(self, ssh):
        """배포 상태 확인"""
        try:
            logger.info("배포 상태 확인 중...")
            
            service_name = self.config["service"]["name"]
            
            # 서비스 상태 확인
            stdin, stdout, stderr = ssh.exec_command(f"systemctl is-active {service_name}")
            status = stdout.read().decode().strip()
            
            if status == "active":
                logger.info("✅ 서비스가 정상적으로 실행 중입니다.")
                
                # 포트 확인
                stdin, stdout, stderr = ssh.exec_command("netstat -tlnp | grep :9000")
                port_output = stdout.read().decode()
                
                if port_output:
                    logger.info("✅ 포트 9000이 정상적으로 열려있습니다.")
                    print(f"🌐 대시보드 접속: http://{self.config['server']['host']}:9000")
                else:
                    logger.warning("⚠️ 포트 9000이 열려있지 않습니다.")
                
                # 로그 확인
                stdin, stdout, stderr = ssh.exec_command(f"journalctl -u {service_name} -n 5 --no-pager")
                logs = stdout.read().decode()
                logger.info(f"최근 로그:\n{logs}")
                
            else:
                logger.error(f"❌ 서비스 상태: {status}")
                
                # 에러 로그 확인
                stdin, stdout, stderr = ssh.exec_command(f"journalctl -u {service_name} -n 10 --no-pager")
                error_logs = stdout.read().decode()
                logger.error(f"에러 로그:\n{error_logs}")
                
                raise Exception(f"서비스가 정상적으로 시작되지 않았습니다: {status}")
            
        except Exception as e:
            logger.error(f"배포 상태 확인 실패: {e}")
            raise
    
    def directory_exists(self, ssh, path):
        """디렉토리 존재 여부 확인"""
        try:
            stdin, stdout, stderr = ssh.exec_command(f"ls -la {path}")
            return stdout.channel.recv_exit_status() == 0
        except:
            return False
    
    def full_deployment(self, commit_message: str = None):
        """전체 배포 프로세스"""
        try:
            print("🚀 AlphaGenesis GitHub 배포 시작")
            print("=" * 50)
            
            # 1. Git 저장소 설정
            print("1️⃣ Git 저장소 설정...")
            self.setup_git_repository()
            
            # 2. 로컬 변경사항 커밋 및 푸시
            print("2️⃣ GitHub에 코드 업로드...")
            if self.commit_and_push(commit_message):
                print("✅ GitHub 업로드 완료")
            else:
                print("ℹ️ 변경사항이 없어 업로드를 건너뜁니다.")
            
            # 3. 서버에 배포
            print("3️⃣ GVS 서버에 배포...")
            self.deploy_to_server()
            
            print("\n🎉 배포 완료!")
            print(f"🌐 대시보드 접속: http://{self.config['server']['host']}:9000")
            print("📊 서비스는 24시간 자동으로 실행됩니다.")
            
        except Exception as e:
            logger.error(f"전체 배포 실패: {e}")
            print(f"❌ 배포 실패: {e}")
            raise

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AlphaGenesis GitHub 배포 도구")
    parser.add_argument("action", choices=["deploy", "push", "pull"], 
                       help="수행할 작업")
    parser.add_argument("-m", "--message", 
                       help="커밋 메시지")
    
    args = parser.parse_args()
    
    deployer = GitHubDeployment()
    
    try:
        if args.action == "deploy":
            deployer.full_deployment(args.message)
        elif args.action == "push":
            deployer.setup_git_repository()
            deployer.commit_and_push(args.message)
        elif args.action == "pull":
            deployer.deploy_to_server()
            
    except Exception as e:
        print(f"❌ 작업 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()