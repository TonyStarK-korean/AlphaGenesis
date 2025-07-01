# 방화벽에서 5001 포트 열기
sudo ufw allow 5001/tcp
sudo ufw reload

# 또는 iptables 사용하는 경우
sudo iptables -A INPUT -p tcp --dport 5001 -j ACCEPT
sudo iptables-save 