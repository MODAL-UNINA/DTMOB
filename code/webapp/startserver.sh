#!/bin/bash

set -ex

python manage.py makemigrations
python -uB manage.py migrate --noinput
python manage.py collectstatic --noinput
gunicorn --config gunicorn_config.py DTMOB_webapp.wsgi:application