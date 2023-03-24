chemprop_train --data_path SMILES-E_gap.csv --dataset_type regression --save_dir LUMO_frzn --split_size 0.8 0.1 0.1 --checkpoint_frzn LUMO_checkpoints/fold_0/model_0/model.pt --dropout 0.2 --target_columns LUMO --epochs 300 --freeze_first_only --gpu 0 --seed 0
chemprop_predict --test_path SMILES-E_gap.csv --checkpoint_dir LUMO_frzn --preds_path LUMO_frzn.csv

chemprop_train --data_path SMILES-E_gap.csv --dataset_type regression --save_dir LUMO_transfer --split_size 0.8 0.1 0.1 --checkpoint_dir LUMO_checkpoints --dropout 0.2 --target_columns LUMO --epochs 70  --gpu 0 --seed 0
chemprop_predict --test_path SMILES-E_gap.csv --checkpoint_dir LUMO_transfer --preds_path LUMO_frzn.csv

chemprop_train --data_path SMILES-E_gap.csv --dataset_type regression --save_dir LUMO --split_size 0.8 0.1 0.1 --dropout 0.2 --target_columns LUMO --epochs 300  --gpu 0 --seed 0
chemprop_predict --test_path SMILES-E_gap.csv --checkpoint_dir LUMO --preds_path LUMO_prediction.csv
