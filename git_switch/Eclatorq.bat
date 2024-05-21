@echo off
set curdir=%~dp0
cd /d %curdir% 
cd ..

call git checkout Eclatorq
echo Eclatorq version :
call git remote
echo branch version :

call git branch 

echo NOW IS Eclatorq version

cmd /k 