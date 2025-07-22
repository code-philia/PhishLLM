@echo off
setlocal ENABLEDELAYEDEXPANSION

REM === CONFIGURATION ===
set "ENV_NAME=phish_env"
set "INSTALL_DIR=%TEMP%\"

REM === FUNCTIONS ===
:success
echo [+] %1
goto :eof

:fail
echo [-] %1
goto :eof

:warn
echo [!] %1
goto :eof

:prompt
set /p yn=%~1 [y/n]:
if /I "%yn%"=="y" (
    exit /b 1
) else if /I "%yn%"=="n" (
    exit /b 0
) else (
    echo Please answer yes or no.
    call :prompt %~1
)
goto :eof

REM === INSTALL CHROME ===
:install_chrome
echo Downloading Chrome...
powershell -Command "Invoke-WebRequest -Uri https://dl.google.com/chrome/install/latest/chrome_installer.exe -OutFile '%INSTALL_DIR%chrome_installer.exe'"
if errorlevel 1 (
    call :fail "Could not download Chrome"
    exit /b 1
)

echo Installing Chrome...
start /wait "" "%INSTALL_DIR%chrome_installer.exe" /silent /install
if errorlevel 1 (
    call :fail "Could not install Chrome"
    exit /b 2
)
call :success "Successfully installed Chrome"
exit /b 0

REM === CHECK CHROME ===
:check_chrome
where chrome >nul 2>&1
if %errorlevel%==0 (
    for /f "delims=" %%v in ('chrome --version') do call :success "Chrome is installed (%%v)"
) else (
    call :warn "Chrome does not seem to be installed"
    call :prompt "Do you want to install Chrome?"
    if %errorlevel%==1 (
        call :install_chrome
    ) else (
        call :fail "Skipping Chrome installation"
    )
)
goto :eof

REM === RUN ===

call :check_chrome

REM Check if Conda environment exists
for /f %%i in ('conda env list ^| findstr /b /c:"%ENV_NAME%"') do (
    call :success "Activating Conda environment %ENV_NAME%"
    goto :env_ready
)

echo Creating Conda environment %ENV_NAME% with Python 3.8
call conda create -y -n %ENV_NAME% python=3.8

:env_ready

REM Install PhishIntention
call conda run -n %ENV_NAME% git clone -b development --single-branch https://github.com/lindsey98/PhishIntention.git
cd PhishIntention
call conda run -n %ENV_NAME% bash setup.sh
cd ..
rmdir /s /q PhishIntention

REM Install requirements.txt
if exist requirements.txt (
    call conda run -n %ENV_NAME% pip install -r requirements.txt
) else (
    echo requirements.txt not found. Skipping.
)

REM Download model using gdown
mkdir checkpoints
cd checkpoints
call conda run -n %ENV_NAME% pip install gdown
call conda run -n %ENV_NAME% gdown --id 1bpy-SRDOkL96j9r3ErBd7L5mDUdLAWaU

echo Done.
pause
