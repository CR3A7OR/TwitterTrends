#!/bin/bash

pip install -r ./requirements.txt

crontab -l | { cat; echo @daily $(which python) $(pwd)/Article_Generator/article_gen.py; } | crontab -
crontab -l | { cat; echo @daily $(which python) $(pwd)/website/webhook.py; } | crontab -
crontab -l | { cat; echo @daily $(which python) $(pwd)/website/emailList.py; } | crontab -

# Uncomment if you want to run straight with uWSGI
#cd ./website
#uwsgi --socket 0.0.0.0:8000 --protocol=http -w wsgi:app
