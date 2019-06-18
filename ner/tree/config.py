import torch

tree_config = {
  "saved": "models/saved_model",
  "data": "training-treebank",
  "emb_dir": "resources/pol/fasttext/",
  "emb_file": "wiki.pl",
  "batchsize": 25,
  "epochs": 5,
  "mem_dim": 300,
  "recurrent_dropout_c": 0.15,
  "recurrent_dropout_h": 0.15,
  "zoneout_choose_child": False,
  "common_mask": False,
  "lr": 0.05,
  "emblr": 0.1,
  "wd": 0.0001,
  "reg": 0.0001,
  "optim": "adagrad",
  "seed": 123,
  "reweight": False,  # 'reweight loss per class to the distrubition of classess in the public dataset'
  "split": 0.1,
  "use_full_training_set": True,
  "cuda":  False or torch.cuda.is_available(),
  "train": False,
  "calculate_new_words": True,
  "input_dim": 50,
  "num_classes": 3,
  "test": True

}