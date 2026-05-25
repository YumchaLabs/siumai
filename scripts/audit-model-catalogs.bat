@echo off
setlocal

pushd "%~dp0\.."
python ".agents\skills\siumai-ai-sdk-maintenance\scripts\audit_model_catalogs.py" --include-green --show-skipped --defer deepinfra %*
set STATUS=%ERRORLEVEL%
popd
exit /b %STATUS%
