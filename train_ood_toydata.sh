#!/bin/bash

# datasets=("blobs" "circles" "moon" "roll")
datasets=("small_circles")

for a in "${datasets[@]}"; do
    python toy_main.py --doc ./WDSM/$a --config ./$a/toydata_ood_wdsm.yml --seed 0 &
    python toy_main.py --doc ./WOEMB/$a --config ./$a/toydata_ood_woemb.yml --seed 0 &
    python toy_main.py --doc ./WONOISE/$a --config ./$a/toydata_ood_ce.yml --seed 0 &
    python toy_main.py --doc ./OURS/$a --config ./$a/toydata_ood.yml --seed 0 &
	wait
	echo "Processing for dataset $a completed."
done