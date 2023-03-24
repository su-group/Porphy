chemprop_train --data_path SMILES-E_gap.csv --dataset_type regression --save_dir HOMO_frzn --split_size 0.8 0.1 0.1 --checkpoint_frzn HOMO_checkpoints/fold_0/model_0/model.pt --dropout 0.2 --target_columns HOMO --epochs 300 --freeze_first_only --gpu 0 --seed 0

chemprop_train --data_path SMILES-E_gap.csv --dataset_type regression --save_dir HOMO_transfer --split_size 0.8 0.1 0.1 --checkpoint_dir HOMO_checkpoints --dropout 0.2 --target_columns HOMO --epochs 300  --gpu 0 --seed 0

chemprop_train --data_path SMILES-E_gap.csv --dataset_type regression --save_dir HOMO --split_size 0.8 0.1 0.1 --dropout 0.2 --target_columns HOMO --epochs 300  --gpu 0 --seed 0

