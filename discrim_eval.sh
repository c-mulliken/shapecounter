#!/bin/bash

for i in {10..90..10}
do
    echo "Evaluating model on mean = $i"
    python eval_discrim.py $i
done