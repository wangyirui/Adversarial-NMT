cd ~/Adversarial-NMT
CUDA_VISIBLE_DEVICES=`free-gpu -n 2` python ~/Adversarial-NMT/joint_train.py --data data-bin/wmt14.en-de/  --joint-batch-size 32 --gpuid 0 --clip-norm 1.0 --beam 1