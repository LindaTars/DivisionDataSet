#!/usr/bin/env bash
# Forzar que el repo root esté en PYTHONPATH para que Python encuentre dataapp
export PYTHONPATH="$(pwd)"
exec gunicorn dataapp.wsgi:application --bind 0.0.0.0:$PORT --log-file -
