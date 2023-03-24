chemprop_train --data_path PBDD.csv --dataset_type regression --save_dir PBDD --split_size 0.8 0.1 0.1  --dropout 0.2 --target_columns E_gap --epochs 300 --gpu 0 
chemprop_predict --test_path PBDD.csv --checkpoint_dir PBDD --preds_path predict_PBDD.csv
chemprop_train --data_path porphyrin.csv --dataset_type regression --save_dir funeturning --split_size 0.8 0.1 0.1 --checkpoint_frzn PBDD/fold_0/model_0/model.pt --dropout 0.2 --target_columns E_gap --epochs 300 --freeze_first_only --gpu 0 --seed 0
chemprop_predict --test_path porphyrin.csv --checkpoint_dir funeturning --preds_path predict_porphyrin.csv
