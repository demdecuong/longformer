#!/bin/bash
# FILES=./cnndm/model/* #cnndm.s2s.transformer.gpu0.epoch22.1
FILES=./cnndm/model/cnndm.s2s.transformer.gpu0.epoch23.3
for f in $FILES; do
    echo "==========================" ${f##*/}
    python -u main.py ${f##*/}
    python prepare_rouge.py
    # cd ./deepmind/result/ 
    # perl /home/pijili/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl -n 4 -w 1.2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0  myROUGE_Config.xml C
    # perl ROUGE-1.5.5.pl -a -c 95 -b 665 -m -n 4 -w 1.2 input.xml > output.xml
    python compute_rouge.py --truth ./cnndm/result/ground_truth/ --summary ./cnndm/result/summary/
    # cd ../../
done
