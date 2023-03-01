@echo off
rem Define here the path to your conda installation
set CONDAPATH=C:\Users\chmelikj\anaconda3
rem Activate the conda environment
call %CONDAPATH%\Scripts\activate.bat
rem Install conda abo environment
call conda env create -f environment.yml
rem Activate conda abo environment
start cmd /k call %CONDAPATH%\Scripts\activate.bat abo
