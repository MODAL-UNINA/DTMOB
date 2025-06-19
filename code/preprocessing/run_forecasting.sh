#!/bin/bash

CONFIG=${1:-"transactions"}

python main_forecasting.py --config=$CONFIG