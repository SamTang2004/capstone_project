@echo off
setlocal

set "REPO_ROOT=%~dp0"
set "VENV_PYTHON=%REPO_ROOT%.venv\Scripts\python.exe"
set "REQUIREMENTS=%REPO_ROOT%requirements.txt"

if not exist "%REQUIREMENTS%" (
    echo ERROR: Missing requirements file: "%REQUIREMENTS%"
    exit /b 1
)

if not exist "%VENV_PYTHON%" (
    echo Creating the validated Python 3.12 virtual environment...
    py -3.12 -m venv "%REPO_ROOT%.venv"
    if errorlevel 1 (
        echo ERROR: Python 3.12 is required to create .venv.
        exit /b 1
    )
)

echo Updating packaging tools without violating supported Torch builds...
"%VENV_PYTHON%" -m pip install --upgrade pip "setuptools<82" wheel
if errorlevel 1 exit /b 1

echo Installing BudgetCUDA requirements excluding torch and torchvision...
"%VENV_PYTHON%" -m pip install --upgrade -r "%REQUIREMENTS%"
if errorlevel 1 exit /b 1

echo Checking the installed dependency set...
"%VENV_PYTHON%" -m pip check
if errorlevel 1 exit /b 1

echo.
echo BudgetCUDA non-Torch requirements are installed.
echo Torch and torchvision were intentionally left unchanged.
"%VENV_PYTHON%" -c "import importlib.util as u; print('torch:', 'installed' if u.find_spec('torch') else 'not installed'); print('torchvision:', 'installed' if u.find_spec('torchvision') else 'not installed')"
echo Install matching torch and torchvision builds separately for the host CUDA version.

endlocal
exit /b 0
