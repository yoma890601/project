set curdir=%~dp0
cd /d %curdir% 
cd ..
call activate Eclatorq

cmd /k @python gui_control.py

#start /min python gui_control.py
exit