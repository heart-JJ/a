@echo off
setlocal
set "EVOAGENT_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%EVOAGENT_PYTHON%" set "EVOAGENT_PYTHON=python"
pushd "%~dp0"
"%EVOAGENT_PYTHON%" -m evoagent %*
set "EVOAGENT_EXIT=%ERRORLEVEL%"
popd
exit /b %EVOAGENT_EXIT%
