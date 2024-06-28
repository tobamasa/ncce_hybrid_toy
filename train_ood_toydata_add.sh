#!/bin/bash

# datasets=("blobs" "circles" "moon" "roll")
datasets=("small_circles_compare")

for a in "${datasets[@]}"; do
    python toy_main.py --doc ./OURS/1 --config ./$a/toydata_ood1.yml --seed 0 &
    python toy_main.py --doc ./OURS/2 --config ./$a/toydata_ood2.yml --seed 0 &
    python toy_main.py --doc ./OURS/3 --config ./$a/toydata_ood2.yml --seed 0 &
    python toy_main.py --doc ./OURS/4 --config ./$a/toydata_ood4.yml --seed 0 &
	# wait
    # python toy_main.py --doc ./OURS/5 --config ./$a/toydata_ood5.yml --seed 0 &
    # python toy_main.py --doc ./OURS/6 --config ./$a/toydata_ood6.yml --seed 0 &
    # python toy_main.py --doc ./OURS/7 --config ./$a/toydata_ood7.yml --seed 0 &
    # python toy_main.py --doc ./OURS/8 --config ./$a/toydata_ood8.yml --seed 0 &
    # wait
    # python toy_main.py --doc ./OURS/9 --config ./$a/toydata_ood9.yml --seed 0 &
    # python toy_main.py --doc ./OURS/10 --config ./$a/toydata_ood10.yml --seed 0 &
    # python toy_main.py --doc ./OURS/11 --config ./$a/toydata_ood11.yml --seed 0 &
	# echo "Processing for dataset $a completed."
done