for i in $(seq 30 10 50); do python3 main.py --perplexity=$i; done
#for (( COUNTER=10; COUNTER<=50; COUNTER+=10 )); do
#    python3 main.py --perplexity=$COUNTER
#
#done