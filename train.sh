source activate py36
python train.py --data data/ --optimizer Adam --learning_rate 1e-3 --lr_shrink_from 8 --model_file data --gpuid 0 --epochs 20