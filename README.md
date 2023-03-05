# Vehicle number plate recognition
Решение тестового задания
### environment
```
git clone git@github.com:caymon14/igs-test-task.git
cd igs-test-task
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```
### run
Скрипт находит положение номеров автомобилей на видео и пытается(не очень успешно) их распознать.
```
python find_numbers.py -v "./data/BMW_320I_2018_black_1.MOV"
```