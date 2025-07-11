@echo off
echo Fixing Git conflict...

REM Fetch remote changes
git fetch origin main

REM Pull with merge strategy
git pull origin main --allow-unrelated-histories

REM If there are conflicts, add all files and commit
git add .
git commit -m "Merge remote and local changes"

REM Push to remote
git push origin main

echo Git conflict resolved!
pause