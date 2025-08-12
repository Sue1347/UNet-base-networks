#!/bin/bash
# This script is used to run training experiments with different parameters.
# nohup sh scripts/run0811.sh &

# about 1.5h
python train.py -e 200 -b 8 -l 1e-3 -exp 001 -p 0.1

python train.py -e 200 -b 8 -l 1e-3 -exp 002 -p 0.2

python evaluate.py -exp 001
python evaluate.py -exp 002

sleep 900

python train.py -e 200 -b 8 -l 1e-3 -exp 003 -p 0.3

python train.py -e 200 -b 8 -l 1e-3 -exp 004 -p 0.4

python evaluate.py -exp 003
python evaluate.py -exp 004

sleep 900

python train.py -e 200 -b 8 -l 1e-3 -exp 005 -p 0.5

python train.py -e 200 -b 8 -l 1e-3 -exp 006 -p 0.6

python evaluate.py -exp 005
python evaluate.py -exp 006

sleep 900



