import argparse

'''
This module defines the command-line arguments for the detection and unlearning framework.

Detection Part: UMAP and HDBSCAN parameters
The clustering parameters are given by the paper: "A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks
https://arxiv.org/pdf/2206.08514"

Unlearning Part: RT:Retrain, GA: Gradient Ascent, NPO: Negative Preference Optimization, RGA:Robust GA

Developer name: Xingyi Zhao
Email: xingyi.zhao@usu.edu
Affiliation: Utah State University [Logan, UT, USA]
Last Modified Time: 2026-02-19

Note: I already set the default values for all the parameters,
      so you can run the code without providing any command-line arguments.
      However, you can modify the default values or provide your own values when running the code.
'''


def get_arguments_bert():
    parser_bert = argparse.ArgumentParser(description="Bert: Backdoor Detection and Unlearning")

    parser_bert.add_argument("--seed", type=list, default=[2021, 2022, 2023])  # Seed for reproducibility
    parser_bert.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    parser_bert.add_argument("--victim_model", type=str, default="bert-base-uncased")
    parser_bert.add_argument("--enable_detection", type=bool, default=True, help="Whether to enable CUBE detection")
   
    parser_bert.add_argument("--dataset", type=str, default="SST-2", help="SST-2, HSOL, AG")
    parser_bert.add_argument("--attack_mode", type=str, default="BadNet", help="BadNet, AddSent, HiddenKiller")

    parser_bert.add_argument("--poisoning_epoch", type=int, default=5)
    parser_bert.add_argument("--max_len", type=int, default=128)
    parser_bert.add_argument("--learning_rate", type=float, default=2e-5)
    parser_bert.add_argument("--batch_size", type=int, default=32)
    
    parser_bert.add_argument("--min_cluster_size", type=int, default=150)
    parser_bert.add_argument("--min_samples", type=int, default=120)
    parser_bert.add_argument("--umap_n_neighbors", type=int, default=100)
    parser_bert.add_argument("--umap_min_dist", type=float, default=0.25)

    parser_bert.add_argument("--unlearning_method", type=str, default="NPO", help="RT, GA, NPO, RGA")
    parser_bert.add_argument("--unlearning_epoch", type=int, default=30)

    return parser_bert

def get_arguments_distilbert():
    parser_distilbert = argparse.ArgumentParser(description="DistilBert: Backdoor Detection and Unlearning")

    parser_distilbert.add_argument("--seed", type=list, default=[2021, 2022, 2023])
    parser_distilbert.add_argument("--tokenizer", type=str, default="distilbert-base-uncased")
    parser_distilbert.add_argument("--victim_model", type=str, default="distilbert-base-uncased")
    parser_distilbert.add_argument("--enable_detection", type=bool, default=True, help="Whether to enable CUBE detection")
    
    parser_distilbert.add_argument("--dataset", type=str, default="SST-2", help="SST-2, HSOL, AG")
    parser_distilbert.add_argument("--attack_mode", type=str, default="BadNet", help="BadNet, AddSent, HiddenKiller")

    parser_distilbert.add_argument("--poisoning_epoch", type=int, default=5)
    parser_distilbert.add_argument("--max_len", type=int, default=128)
    parser_distilbert.add_argument("--learning_rate", type=float, default=2e-5)
    parser_distilbert.add_argument("--batch_size", type=int, default=32)

    parser_distilbert.add_argument("--min_cluster_size", type=int, default=150)
    parser_distilbert.add_argument("--min_samples", type=int, default=120)
    parser_distilbert.add_argument("--umap_n_neighbors", type=int, default=100)
    parser_distilbert.add_argument("--umap_min_dist", type=float, default=0.15)  # 0.15 for more compact clusters -- distilbert

    parser_distilbert.add_argument("--unlearning_method", type=str, default="GA", help="RT, GA, NPO, RGA")
    parser_distilbert.add_argument("--unlearning_epoch", type=int, default=30)

    return parser_distilbert

def get_arguments_llama2():
    parser_llama = argparse.ArgumentParser(description="Llama2-7B: Backdoor Detection and Unlearning")

    parser_llama.add_argument("--seed", type=list, default=[2021, 2022, 2023])
    parser_llama.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")
    parser_llama.add_argument("--victim_model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser_llama.add_argument("--enable_detection", type=bool, default=True, help="Whether to enable CUBE detection")

    parser_llama.add_argument("--dataset", type=str, default="SST-2", help="SST-2, HSOL, AG")
    parser_llama.add_argument("--attack_mode", type=str, default="HiddenKiller", help="BadNet, AddSent, HiddenKiller")

    parser_llama.add_argument("--poisoning_epoch", type=int, default=5)
    parser_llama.add_argument("--max_len", type=int, default=128)
    parser_llama.add_argument("--learning_rate", type=float, default=5e-6)
    parser_llama.add_argument("--batch_size", type=int, default=32)

    parser_llama.add_argument("--min_cluster_size", type=int, default=150)
    parser_llama.add_argument("--min_samples", type=int, default=120)
    parser_llama.add_argument("--umap_n_neighbors", type=int, default=100)
    parser_llama.add_argument("--umap_min_dist", type=float, default=0.25)

    parser_llama.add_argument("--unlearning_method", type=str, default="NPO", help="RT, GA, NPO, RGA")
    parser_llama.add_argument("--unlearning_epoch", type=int, default=30)

    return parser_llama
