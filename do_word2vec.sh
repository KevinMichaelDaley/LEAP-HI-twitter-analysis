while true; do
    export PATH=$PATH:/usr/local/cuda-10.0:/usr/local/cuda-10.0/bin
    python3 use_word2vec.py --nspass tweat 127.0.0.1 tweat gun_sample
done
