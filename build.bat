@echo off
echo Installing PyInstaller...
pip install pyinstaller

echo Building PickHero...
pyinstaller pickhero.spec --noconfirm

echo.
echo Build complete. Output in dist\PickHero.exe
echo Make sure to place a "songs" folder next to the exe.
pause
