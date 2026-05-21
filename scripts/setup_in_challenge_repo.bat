@echo off
setlocal

if "%~1"=="" (
  echo usage: scripts\setup_in_challenge_repo.bat C:\path\to\macro-place-challenge-2026
  exit /b 2
)

set "CHALLENGE_REPO=%~1"
if not exist "%CHALLENGE_REPO%\submissions" (
  echo error: expected a Partcl/HRT macro-place-challenge-2026 checkout
  exit /b 2
)
if not exist "%CHALLENGE_REPO%\macro_place" (
  echo error: expected a Partcl/HRT macro-place-challenge-2026 checkout
  exit /b 2
)

if not exist "%CHALLENGE_REPO%\submissions\retryoos" mkdir "%CHALLENGE_REPO%\submissions\retryoos"
copy /Y placer.py "%CHALLENGE_REPO%\submissions\v57_soft_overlap_sa.py" >nul
copy /Y submissions\__init__.py "%CHALLENGE_REPO%\submissions\__init__.py" >nul
copy /Y submissions\retryoos\__init__.py "%CHALLENGE_REPO%\submissions\retryoos\__init__.py" >nul
copy /Y submissions\retryoos\top1_incremental_sa.py "%CHALLENGE_REPO%\submissions\retryoos\top1_incremental_sa.py" >nul
copy /Y submissions\retryoos\top1_soft_overlap_sa.py "%CHALLENGE_REPO%\submissions\retryoos\top1_soft_overlap_sa.py" >nul
copy /Y submissions\retryoos\top1_replace_sa.py "%CHALLENGE_REPO%\submissions\retryoos\top1_replace_sa.py" >nul

echo Installed V57 Soft-Overlap SA placer into: %CHALLENGE_REPO%\submissions\v57_soft_overlap_sa.py
echo Run: cd /d "%CHALLENGE_REPO%" ^&^& uv run evaluate submissions/v57_soft_overlap_sa.py --all
