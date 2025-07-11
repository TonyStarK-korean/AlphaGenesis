#!/usr/bin/env python3
"""
GVS 서버 배포 자동화 스크립트
로컬 파일을 GVS 서버로 자동 업로드 및 배포
"""

import os
import sys
import subprocess
import shutil
import zipfile
import json
from datetime import datetime
from pathlib import Path
import paramiko
from scp import SCPClient
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GVSDeployment:
    """GVS 서버 배포 클래스"""
    
    def __init__(self, config_file: str = "deployment_config.json"):
        """
        초기화
        
        Args:
            config_file: 배포 설정 파일 경로
        """
        self.config = self.load_config(config_file)
        self.project_root = Path(__file__).parent.parent
        self.deployment_dir = self.project_root / "deployment"
        self.temp_dir = self.deployment_dir / "temp"
        
        # 서버 연결 정보
        self.server_host = self.config.get('server', {}).get('host', '34.47.77.230')
        self.server_port = self.config.get('server', {}).get('port', 22)
        self.server_user = self.config.get('server', {}).get('user', 'root')
        self.server_password = self.config.get('server', {}).get('password', '')
        self.server_key_file = self.config.get('server', {}).get('key_file', '')
        
        # 서버 경로
        self.remote_project_dir = self.config.get('paths', {}).get('remote_project_dir', '/root/alphagenesis')
        self.remote_backup_dir = self.config.get('paths', {}).get('remote_backup_dir', '/root/backups')
        
        logger.info(f"GVS 배포 시스템 초기화: {self.server_host}:{self.server_port}")
    
    def load_config(self, config_file: str) -> dict:
        """
        배포 설정 파일 로드
        
        Args:
            config_file: 설정 파일 경로
            
        Returns:
            설정 딕셔너리
        """
        try:
            config_path = Path(__file__).parent / config_file
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 기본 설정 생성
                default_config = {
                    "server": {
                        "host": "34.47.77.230",
                        "port": 22,
                        "user": "root",
                        "password": "",
                        "key_file": ""
                    },
                    "paths": {
                        "remote_project_dir": "/root/alphagenesis",
                        "remote_backup_dir": "/root/backups"
                    },
                    "exclude_patterns": [
                        "*.pyc",
                        "__pycache__",
                        ".git",
                        "*.log",
                        "temp",
                        "deployment/temp",
                        "results",
                        "logs/*.log"
                    ],
                    "required_packages": [
                        "flask",
                        "flask-cors",
                        "pandas",
                        "numpy",
                        "ccxt",
                        "scikit-learn",
                        "xgboost",
                        "paramiko",
                        "scp"
                    ]
                }
                
                # 기본 설정 파일 생성
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                
                logger.info(f"기본 설정 파일 생성: {config_path}")
                return default_config
                
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            return {}
    
    def create_ssh_connection(self) -> paramiko.SSHClient:
        """
        SSH 연결 생성
        
        Returns:
            SSH 클라이언트 객체
        """
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 키 파일 또는 패스워드 인증
            if self.server_key_file and os.path.exists(self.server_key_file):
                ssh.connect(
                    hostname=self.server_host,
                    port=self.server_port,
                    username=self.server_user,
                    key_filename=self.server_key_file
                )
                logger.info("SSH 키 파일 인증 성공")
            elif self.server_password:
                ssh.connect(
                    hostname=self.server_host,
                    port=self.server_port,
                    username=self.server_user,
                    password=self.server_password
                )
                logger.info("SSH 패스워드 인증 성공")
            else:
                raise Exception("SSH 인증 정보가 없습니다.")
            
            return ssh
            
        except Exception as e:
            logger.error(f"SSH 연결 실패: {e}")
            raise
    
    def create_deployment_package(self) -> str:
        """
        배포 패키지 생성
        
        Returns:
            배포 패키지 파일 경로
        """
        try:
            # 임시 디렉토리 생성
            self.temp_dir.mkdir(exist_ok=True)
            
            # 배포 패키지 파일명
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            package_name = f"alphagenesis_deploy_{timestamp}.zip"
            package_path = self.temp_dir / package_name
            
            # 제외 패턴
            exclude_patterns = self.config.get('exclude_patterns', [])
            
            logger.info("배포 패키지 생성 중...")
            
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.project_root):
                    # 제외 디렉토리 필터링
                    dirs[:] = [d for d in dirs if not any(
                        self._match_pattern(d, pattern) for pattern in exclude_patterns
                    )]
                    
                    for file in files:
                        # 제외 파일 필터링
                        if any(self._match_pattern(file, pattern) for pattern in exclude_patterns):
                            continue
                        
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.project_root)
                        zipf.write(file_path, arcname)
            
            logger.info(f"배포 패키지 생성 완료: {package_path}")
            return str(package_path)
            
        except Exception as e:
            logger.error(f"배포 패키지 생성 실패: {e}")
            raise
    
    def _match_pattern(self, name: str, pattern: str) -> bool:
        """
        패턴 매칭 검사
        
        Args:
            name: 검사할 이름
            pattern: 패턴
            
        Returns:
            매칭 여부
        """
        import fnmatch
        return fnmatch.fnmatch(name, pattern)
    
    def backup_remote_project(self, ssh: paramiko.SSHClient):
        """
        원격 프로젝트 백업
        
        Args:
            ssh: SSH 클라이언트
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"alphagenesis_backup_{timestamp}.tar.gz"
            backup_path = f"{self.remote_backup_dir}/{backup_name}"
            
            # 백업 디렉토리 생성
            ssh.exec_command(f"mkdir -p {self.remote_backup_dir}")
            
            # 기존 프로젝트 백업
            if self.remote_project_exists(ssh):
                backup_cmd = f"cd {os.path.dirname(self.remote_project_dir)} && tar -czf {backup_path} {os.path.basename(self.remote_project_dir)}"
                stdin, stdout, stderr = ssh.exec_command(backup_cmd)
                
                if stderr.read():
                    logger.warning(f"백업 중 경고: {stderr.read().decode()}")
                
                logger.info(f"원격 프로젝트 백업 완료: {backup_path}")
            
        except Exception as e:
            logger.error(f"원격 프로젝트 백업 실패: {e}")
    
    def remote_project_exists(self, ssh: paramiko.SSHClient) -> bool:
        """
        원격 프로젝트 존재 여부 확인
        
        Args:
            ssh: SSH 클라이언트
            
        Returns:
            존재 여부
        """
        try:
            stdin, stdout, stderr = ssh.exec_command(f"ls -la {self.remote_project_dir}")
            return stdout.channel.recv_exit_status() == 0
        except:
            return False
    
    def upload_package(self, ssh: paramiko.SSHClient, package_path: str):
        """
        패키지 업로드
        
        Args:
            ssh: SSH 클라이언트
            package_path: 패키지 파일 경로
        """
        try:
            # 원격 디렉토리 생성
            ssh.exec_command(f"mkdir -p {os.path.dirname(self.remote_project_dir)}")
            
            # 파일 업로드
            with SCPClient(ssh.get_transport()) as scp:
                remote_package = f"/tmp/{os.path.basename(package_path)}"
                scp.put(package_path, remote_package)
                logger.info(f"패키지 업로드 완료: {remote_package}")
            
            # 기존 프로젝트 삭제 (백업 후)
            if self.remote_project_exists(ssh):
                ssh.exec_command(f"rm -rf {self.remote_project_dir}")
            
            # 새 프로젝트 디렉토리 생성
            ssh.exec_command(f"mkdir -p {self.remote_project_dir}")
            
            # 패키지 압축 해제
            extract_cmd = f"cd {self.remote_project_dir} && unzip -q {remote_package}"
            stdin, stdout, stderr = ssh.exec_command(extract_cmd)
            
            if stderr.read():
                logger.error(f"압축 해제 실패: {stderr.read().decode()}")
                raise Exception("압축 해제 실패")
            
            # 임시 파일 정리
            ssh.exec_command(f"rm -f {remote_package}")
            
            logger.info("패키지 압축 해제 완료")
            
        except Exception as e:
            logger.error(f"패키지 업로드 실패: {e}")
            raise
    
    def install_dependencies(self, ssh: paramiko.SSHClient):
        """
        의존성 설치
        
        Args:
            ssh: SSH 클라이언트
        """
        try:
            logger.info("의존성 설치 중...")
            
            # pip 업그레이드
            ssh.exec_command("pip install --upgrade pip")
            
            # requirements.txt 확인
            requirements_path = f"{self.remote_project_dir}/requirements.txt"
            stdin, stdout, stderr = ssh.exec_command(f"ls -la {requirements_path}")
            
            if stdout.channel.recv_exit_status() == 0:
                # requirements.txt로 설치
                install_cmd = f"cd {self.remote_project_dir} && pip install -r requirements.txt"
                stdin, stdout, stderr = ssh.exec_command(install_cmd)
                
                # 실행 결과 확인
                output = stdout.read().decode()
                error = stderr.read().decode()
                
                if error:
                    logger.warning(f"의존성 설치 경고: {error}")
                
                logger.info("requirements.txt 기반 의존성 설치 완료")
            else:
                # 개별 패키지 설치
                packages = self.config.get('required_packages', [])
                for package in packages:
                    install_cmd = f"pip install {package}"
                    stdin, stdout, stderr = ssh.exec_command(install_cmd)
                    
                    if stderr.read():
                        logger.warning(f"{package} 설치 실패")
                    else:
                        logger.info(f"{package} 설치 완료")
            
            logger.info("의존성 설치 완료")
            
        except Exception as e:
            logger.error(f"의존성 설치 실패: {e}")
    
    def setup_systemd_service(self, ssh: paramiko.SSHClient):
        """
        systemd 서비스 설정
        
        Args:
            ssh: SSH 클라이언트
        """
        try:
            logger.info("systemd 서비스 설정 중...")
            
            # 서비스 파일 내용
            service_content = f"""[Unit]
Description=AlphaGenesis Trading Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={self.remote_project_dir}
Environment=PYTHONPATH={self.remote_project_dir}
ExecStart=/usr/bin/python3 {self.remote_project_dir}/dashboard/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
            
            # 서비스 파일 생성
            service_path = "/etc/systemd/system/alphagenesis.service"
            
            # 임시 파일로 서비스 파일 생성
            temp_service = "/tmp/alphagenesis.service"
            stdin, stdout, stderr = ssh.exec_command(f"cat > {temp_service} << 'EOF'\n{service_content}EOF")
            
            # 서비스 파일 이동
            ssh.exec_command(f"mv {temp_service} {service_path}")
            
            # systemd 리로드
            ssh.exec_command("systemctl daemon-reload")
            
            # 서비스 활성화
            ssh.exec_command("systemctl enable alphagenesis.service")
            
            logger.info("systemd 서비스 설정 완료")
            
        except Exception as e:
            logger.error(f"systemd 서비스 설정 실패: {e}")
    
    def start_service(self, ssh: paramiko.SSHClient):
        """
        서비스 시작
        
        Args:
            ssh: SSH 클라이언트
        """
        try:
            logger.info("서비스 시작 중...")
            
            # 기존 서비스 중지
            ssh.exec_command("systemctl stop alphagenesis.service")
            
            # 서비스 시작
            stdin, stdout, stderr = ssh.exec_command("systemctl start alphagenesis.service")
            
            # 서비스 상태 확인
            stdin, stdout, stderr = ssh.exec_command("systemctl is-active alphagenesis.service")
            status = stdout.read().decode().strip()
            
            if status == "active":
                logger.info("서비스 시작 완료")
            else:
                logger.error(f"서비스 시작 실패: {status}")
                
                # 에러 로그 확인
                stdin, stdout, stderr = ssh.exec_command("journalctl -u alphagenesis.service -n 10")
                error_logs = stdout.read().decode()
                logger.error(f"서비스 에러 로그:\n{error_logs}")
            
        except Exception as e:
            logger.error(f"서비스 시작 실패: {e}")
    
    def configure_firewall(self, ssh: paramiko.SSHClient):
        """
        방화벽 설정
        
        Args:
            ssh: SSH 클라이언트
        """
        try:
            logger.info("방화벽 설정 중...")
            
            # UFW 설치 및 활성화
            ssh.exec_command("apt-get update && apt-get install -y ufw")
            
            # 포트 9000 허용
            ssh.exec_command("ufw allow 9000/tcp")
            
            # SSH 포트 허용 (안전을 위해)
            ssh.exec_command("ufw allow ssh")
            
            # 방화벽 활성화
            ssh.exec_command("ufw --force enable")
            
            logger.info("방화벽 설정 완료")
            
        except Exception as e:
            logger.error(f"방화벽 설정 실패: {e}")
    
    def deploy(self):
        """
        전체 배포 프로세스 실행
        """
        try:
            logger.info("🚀 GVS 서버 배포 시작")
            
            # 1. 배포 패키지 생성
            package_path = self.create_deployment_package()
            
            # 2. SSH 연결
            ssh = self.create_ssh_connection()
            
            try:
                # 3. 원격 프로젝트 백업
                self.backup_remote_project(ssh)
                
                # 4. 패키지 업로드
                self.upload_package(ssh, package_path)
                
                # 5. 의존성 설치
                self.install_dependencies(ssh)
                
                # 6. systemd 서비스 설정
                self.setup_systemd_service(ssh)
                
                # 7. 방화벽 설정
                self.configure_firewall(ssh)
                
                # 8. 서비스 시작
                self.start_service(ssh)
                
                logger.info("🎉 배포 완료!")
                logger.info(f"📊 대시보드 접속: http://{self.server_host}:9000")
                
            finally:
                ssh.close()
                
                # 9. 임시 파일 정리
                if os.path.exists(package_path):
                    os.remove(package_path)
                    logger.info("임시 파일 정리 완료")
                
        except Exception as e:
            logger.error(f"❌ 배포 실패: {e}")
            raise
    
    def check_status(self):
        """
        서버 상태 확인
        """
        try:
            logger.info("서버 상태 확인 중...")
            
            ssh = self.create_ssh_connection()
            
            try:
                # 서비스 상태 확인
                stdin, stdout, stderr = ssh.exec_command("systemctl status alphagenesis.service")
                status_output = stdout.read().decode()
                
                print("📊 서비스 상태:")
                print(status_output)
                
                # 포트 확인
                stdin, stdout, stderr = ssh.exec_command("netstat -tlnp | grep :9000")
                port_output = stdout.read().decode()
                
                if port_output:
                    print("✅ 포트 9000 정상 동작 중")
                else:
                    print("❌ 포트 9000 비활성화")
                
                # 시스템 리소스 확인
                stdin, stdout, stderr = ssh.exec_command("free -h && df -h")
                resource_output = stdout.read().decode()
                
                print("💻 시스템 리소스:")
                print(resource_output)
                
            finally:
                ssh.close()
                
        except Exception as e:
            logger.error(f"상태 확인 실패: {e}")
    
    def restart_service(self):
        """
        서비스 재시작
        """
        try:
            logger.info("서비스 재시작 중...")
            
            ssh = self.create_ssh_connection()
            
            try:
                # 서비스 재시작
                stdin, stdout, stderr = ssh.exec_command("systemctl restart alphagenesis.service")
                
                # 상태 확인
                stdin, stdout, stderr = ssh.exec_command("systemctl is-active alphagenesis.service")
                status = stdout.read().decode().strip()
                
                if status == "active":
                    logger.info("✅ 서비스 재시작 완료")
                else:
                    logger.error(f"❌ 서비스 재시작 실패: {status}")
                
            finally:
                ssh.close()
                
        except Exception as e:
            logger.error(f"서비스 재시작 실패: {e}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GVS 서버 배포 도구")
    parser.add_argument("action", choices=["deploy", "status", "restart"], 
                       help="수행할 작업")
    parser.add_argument("--config", default="deployment_config.json", 
                       help="설정 파일 경로")
    
    args = parser.parse_args()
    
    # 배포 객체 생성
    deployer = GVSDeployment(args.config)
    
    # 작업 실행
    if args.action == "deploy":
        deployer.deploy()
    elif args.action == "status":
        deployer.check_status()
    elif args.action == "restart":
        deployer.restart_service()

if __name__ == "__main__":
    main()