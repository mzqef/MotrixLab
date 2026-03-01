@echo off
cd /d D:\MotrixLab
uv run python _resume_s9.py > _s9_stdout.txt 2> _s9_stderr.txt
echo S9 finished with exit code %ERRORLEVEL% >> _s9_stdout.txt
