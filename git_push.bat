@echo off
echo Checking git status...
git status

echo.
echo Adding all files...
git add .

echo.
echo Checking status after add...
git status

echo.
echo Committing changes...
git commit -m "Full model integration with signals dashboard"

echo.
echo Pushing to remote...
git push Kalshicommandcenter main

echo.
echo Done!
pause
