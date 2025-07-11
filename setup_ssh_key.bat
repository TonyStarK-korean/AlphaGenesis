@echo off
echo Setting up SSH key for AlphaGenesis...

REM Generate SSH key
echo Generating SSH key...
ssh-keygen -t rsa -b 4096 -C "alphagenesis@server" -f %USERPROFILE%\.ssh\id_rsa -N ""

REM Display public key
echo.
echo Your public key (copy this to server):
echo ============================================
type %USERPROFILE%\.ssh\id_rsa.pub
echo ============================================
echo.

REM Update config file
echo Updating config file...
powershell -Command "(Get-Content deployment\git_config.json) -replace '\"key_file\": \"\"', '\"key_file\": \"%USERPROFILE%\.ssh\id_rsa\"' -replace '\"password\": \"[^\"]*\"', '\"password\": \"\"' | Set-Content deployment\git_config.json"

echo.
echo SSH key setup completed!
echo Please add the public key above to your server's ~/.ssh/authorized_keys file
echo.
pause