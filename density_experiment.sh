#!/bin/bash

for i in {10..90..10}
do
    echo "Training model on mean = $i"
    python train.py $i
done