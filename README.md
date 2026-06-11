LPIMatch

保留的主实验配置为 `--label_pi_mode off --candidate_pi_mode association --association_backend transformer_mask`。

参考运行命令：

```powershell
python .\run.py --dataset_name aids --label_pi_mode off --candidate_pi_mode association --association_backend transformer_mask --association_heads 4 --candidate_maxloop 10 --name assoc_transformer_mask
```

使用 GRF 数据目录时，`--dataset_name` 指向包含 `queries` 和 `targets` 的目录：

```powershell
python .\run.py --dataset_name D:\path\to\query_size4_hop2_500 --label_pi_mode off --candidate_pi_mode association --association_backend transformer_mask --association_heads 4 --candidate_maxloop 10 --candidate_eval_batch_size 1 --name assoc_transformer_mask
```
