@echo off
set curdir=%~dp0
cd /d %curdir% 
call conda create -n Eclatorq python=3.9 -y 

call activate Eclatorq

pip install -r requirements-gpu.txt 
pip install -r requirements.txt
pip install -r all_requirements.txt
echo all_requirements.txt
@echo off
pip uninstall keras>=2.5.0 -y
pip install typing_extensions==4.9.0

pip install mealpy==3.0.1
pip install niapy==2.1.0
pip install numpy==1.23.0

cd ..
cmd /k @python gui_control.py