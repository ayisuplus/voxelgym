@echo off
REM Build voxelgym_rs into the project venv (release).
set "PATH=C:\Users\ayi.dnk\.cargo\bin;%PATH%"
set "VIRTUAL_ENV=D:\projects\AI-minecraft\.venv"
cd /d D:\projects\AI-minecraft
.venv\Scripts\maturin.exe develop --release -m crates\voxel-py\Cargo.toml
