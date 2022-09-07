@echo off && color 0e && cls

set entry_file=main.c
set bin_path=bin
set bin_out_file=main.exe

echo __ cleaning

if exist %bin_path% rmdir /s /q %bin_path%
mkdir %bin_path%> nul

echo __ compiling

clang %entry_file% -o %bin_path%/%bin_out_file%
if not exist %bin_path%/%bin_out_file% set error=Missing executable file %bin_path%/%bin_out_file% && goto :error

echo __ running

call %bin_path%\%bin_out_file%

echo __ exiting && echo.

:crash
    if defined error color 0c && echo %error% && exit
    
color 0a && exit
