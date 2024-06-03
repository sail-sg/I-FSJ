#!/bin/bash -i
time python -u exps/rs.py --user_prompt_index $1 --shots $2 > ./logs/rs_$2/$1.txt 2>&1