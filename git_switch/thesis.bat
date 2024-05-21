@echo off
set curdir=%~dp0
cd /d %curdir% 
cd ..

call git checkout thesis
echo thesis version :
call git remote
echo branch version :

call git branch 

echo NOW IS thesis version

cmd /k 