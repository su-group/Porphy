chemprop_train --data_path porphyrin.csv --dataset_type regression --save_dir self-train --split_size 0.8 0.1 0.1 --dropout 0.1 --target_columns E_gap --epochs 300
chemprop_predict --test_path porphyrin.csv --checkpoint_dir self-train --preds_path predict_self.csv
