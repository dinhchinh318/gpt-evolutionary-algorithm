@echo off
setlocal

echo ===============================
echo RUN SCENARIO 1
echo ===============================

set JAR=io.github.ericmedvet.jgea.experimenter\target\jgea.experimenter-2.8.1-jar-with-dependencies.jar
set CONFIG=scenario1.txt

if not exist "%JAR%" (
    echo KHONG TIM THAY JAR:
    echo %JAR%
    pause
    exit /b 1
)

if not exist "%CONFIG%" (
    echo KHONG TIM THAY CONFIG:
    echo %CONFIG%
    pause
    exit /b 1
)

java -Xmx24g -jar "%JAR%" -f "%CONFIG%"

echo.
echo DONE
pause
