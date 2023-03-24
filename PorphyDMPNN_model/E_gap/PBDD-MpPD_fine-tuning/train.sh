chemprop_train --data_path porphyrin.csv --dataset_type regression --save_dir transfer --split_size 0.8 0.1 0.1 --checkpoint_dir PBDD --dropout 0.2 --target_columns E_gap --epochs 300  --gpu 0 --seed 0
chemprop_predict --test_path porphyrin.csv --checkpoint_dir transfer --preds_path transfer_predict.csv
