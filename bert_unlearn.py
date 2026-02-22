import torch
import copy
import umap
import gc
import hdbscan
import pandas as pd
import numpy as np
from torch import nn as nn
from config import get_arguments_bert
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from util import set_seed, TargetDataset, BertClassification, train, evaluate, rep_extract_bert, ga_train, npo_train, rga_train, compute_logits_ref

"""
This modules implements the backdoor attack and unlearning process for Bert-based text classification models. It includes the following steps:
1. Training a backdoored Bert model on a poisoned dataset.
2. Evaluating the backdoored model on both clean and poisoned test sets.
3. (Optional) Detecting the poisoned samples using CUBE, which consists of UMAP
    for dimension reduction and HDBSCAN for clustering.
4. Unlearning the backdoor using one of the following methods: Retraining (RT), Gradient Ascent (GA), Negative Preference Optimization (NPO), and Robust Gradient Ascent (RGA).

Note: We test the code on different GPUs including H200 [1 GPU], A6000ada [4 GPUs] and 3090 [one 3090 only for bert and distilbert]; The results of NPO could be different [But you can still view the poison loss keeps increasing in GA and NPO].
      The increasing poison loss would eventually lead to trigger shifting.
      However, RGA is more stable and consistently outperforms NPO in most cases. 

Developer name: Xingyi Zhao
Email: xingyi.zhao@usu.edu
Affiliation: Utah State University [Logan, UT, USA]
Last Modified Time: 2026-02-19
"""


if __name__ == '__main__':
    args = get_arguments_bert().parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for current_seed in args.seed:
        print(f"\n================= Current Seed: {current_seed} =================")
        print("***************** Training the Backdoor Model *****************")
        set_seed(current_seed)

        # Dataset and Dataloader
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

        train_df_poisoned = pd.read_csv(f"Data_Poisoning/{args.attack_mode}/{args.dataset}/train_poisoned.csv")
        train_df_clean = pd.read_csv(f"Data_Poisoning/{args.attack_mode}/{args.dataset}/train_clean.csv")

        test_df_clean = pd.read_csv(f"Data_Poisoning/{args.attack_mode}/{args.dataset}/test_clean.csv")
        test_df_poisoned = pd.read_csv(f"Data_Poisoning/{args.attack_mode}/{args.dataset}/test_poisoned_all.csv")

        train_loader_poisoned = DataLoader(TargetDataset(tokenizer=tokenizer, max_len=args.max_len, data=train_df_poisoned), batch_size=args.batch_size, shuffle=False)
        train_loader_clean = DataLoader(TargetDataset(tokenizer=tokenizer, max_len=args.max_len, data=train_df_clean), batch_size=args.batch_size, shuffle=False)

        test_loader_clean = DataLoader(TargetDataset(tokenizer=tokenizer, max_len=args.max_len, data=test_df_clean), batch_size=args.batch_size, shuffle=False)
        test_loader_poisoned = DataLoader(TargetDataset(tokenizer=tokenizer, max_len=args.max_len, data=test_df_poisoned), batch_size=args.batch_size, shuffle=False)

        print("--------------- Dataset Statistics ---------------")
        clean_df = train_df_poisoned[train_df_poisoned["poisoned"] == 0].reset_index(drop=True)
        poisoned_df = train_df_poisoned[train_df_poisoned["poisoned"] == 1].reset_index(drop=True)

        print(f"Dataset: {args.dataset}")
        print(f"Clean Dataset Size: {len(clean_df)}")
        print(f"Poisoned Dataset Size: {len(poisoned_df)}")

        if args.dataset == "SST-2":
            print(f"Clean Positive Samples: {len(clean_df[clean_df['label'] == 1])}")
            print(f"Clean Negative Samples: {len(clean_df[clean_df['label'] == 0])}")

        elif args.dataset == "HSOL":
            print(f"Clean Non-toxic Samples: {len(clean_df[clean_df['label'] == 0])}")
            print(f"Clean Toxic Samples: {len(clean_df[clean_df['label'] == 1])}")

        elif args.dataset == "AG":
            print(f"Clean World Samples: {len(clean_df[clean_df['label'] == 0])}")
            print(f"Clean Sports Samples: {len(clean_df[clean_df['label'] == 1])}")
            print(f"Clean Business Samples: {len(clean_df[clean_df['label'] == 2])}")
            print(f"Clean Science Samples: {len(clean_df[clean_df['label'] == 3])}")

        else:
            raise ValueError("Invalid dataset")

        # Load the Bert model
        num_class = 4 if args.dataset == "AG" else 2
        target_model = BertClassification(BertModel.from_pretrained(args.victim_model), label_num=num_class).to(device)

        print("--------------- Poisoning Phase ---------------")
        for epoch in range(args.poisoning_epoch):
            print("\n-------------------")
            train_loss = train(target_model, train_loader_poisoned, args, device)
            print(f"Epoch {epoch + 1}/{args.poisoning_epoch} | Train loss {train_loss}")

        print("-------------------------------------------------")
        print("1. Clean Performance: Evaluation on Clean Test Set:")
        evaluate(target_model, test_loader_clean, args.dataset, device)

        print("2. Poisoning Performance: Evaluation on Poisoned (ALL) Test Set:")
        evaluate(target_model, test_loader_poisoned, args.dataset, device)
        print("-------------------------------------------------")

        if args.enable_detection:

            print("***************** Poisoned Samples Detection (CUBE) *****************")
            print("--------------- Step1: Dimension Reduction (UMAP) ---------------")
            clean_loader_visualize = DataLoader(TargetDataset(tokenizer, args.max_len, clean_df), batch_size=args.batch_size, shuffle=False)
            poisoned_loader_visualize = DataLoader(TargetDataset(tokenizer, args.max_len, poisoned_df), batch_size=args.batch_size, shuffle=False)

            poisoned_rep = rep_extract_bert(target_model.bert, poisoned_loader_visualize, device)  # Extract poisoned rep
            clean_rep = rep_extract_bert(target_model.bert, clean_loader_visualize, device)  # Extract clean rep
            overall_rep = np.concatenate((poisoned_rep, clean_rep), axis=0)  # Overall rep

            reducer = umap.UMAP(n_components=4,
                                n_neighbors=args.umap_n_neighbors,
                                min_dist=args.umap_min_dist,
                                metric="cosine",
                                random_state=current_seed
                                )
            projected_rep = reducer.fit_transform(overall_rep)

            print("--------------- Step2: Clustering (HDBSCAN) ---------------")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                                        min_samples=args.min_samples,
                                        metric="euclidean").fit(projected_rep)
            labels = clusterer.labels_

            cnt = Counter(labels)
            num_clusters = 4 if args.dataset == "AG" else 2
            clean_cids = [cid for cid, _ in cnt.most_common() if cid != -1][:num_clusters]
            pred_clean_mask = np.isin(labels, clean_cids)
            pred_poison_mask = ~pred_clean_mask

            true_poison = np.concatenate([
                np.ones(len(poisoned_df), dtype=bool),
                np.zeros(len(clean_df), dtype=bool)
            ])

            prec, rec, f1, _ = precision_recall_fscore_support(true_poison, pred_poison_mask, average="binary")
            print(f" Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

            # Uncomment the following code to visualize the clustering results (UMAP projection)
            # print("--------------- Visualization ---------------")
            # poisoned_rep_2 = projected_rep[:len(poisoned_df)]
            # clean_rep_2 = projected_rep[len(poisoned_df):]
            #
            # plt.figure(figsize=(10, 8))
            # ax = plt.gca()
            # for s in ["top", "bottom", "left", "right"]:
            #     ax.spines[s].set_linewidth(4)
            #
            # plt.xticks(fontsize=30)
            # plt.yticks(fontsize=30)
            # plt.tick_params(axis='both', labelsize=30, width=4, length=10)
            #
            # plt.scatter(poisoned_rep_2[:, 0], poisoned_rep_2[:, 1], s=20, c=(0, 0.5, 0), label="Poisoned", alpha=0.5, marker="o")
            # plt.scatter(clean_rep_2[:, 0], clean_rep_2[:, 1], s=20, c=(0.5, 0.2, 0.8), label="Clean", alpha=0.5, marker="o")
            # plt.legend(frameon=False, fontsize=25, markerscale=3)
            # plt.tight_layout()
            # plt.savefig(f"visualization_{args.dataset}_{args.attack_mode}_{current_seed}.png")
            # plt.show()
            # plt.close()

            mask_poison_tbl = pd.Series(pred_poison_mask)
            mask_clean_tbl = ~mask_poison_tbl

            poisoned_df_detected = pd.concat([
                poisoned_df[mask_poison_tbl.iloc[:len(poisoned_df)].values],
                clean_df[mask_poison_tbl.iloc[len(poisoned_df):].values]
            ]).reset_index(drop=True)

            clean_df_detected = pd.concat([
                poisoned_df[mask_clean_tbl.iloc[:len(poisoned_df)].values],
                clean_df[mask_clean_tbl.iloc[len(poisoned_df):].values]
            ]).reset_index(drop=True)

            print(f"Detected Poisoned Samples: {len(poisoned_df_detected)}")
            print(f"Detected Clean Samples: {len(clean_df_detected)}")
        
        else:  # If detection is not enabled, we assume the clean samples are correctly detected
            clean_df_detected = clean_df 
            poisoned_df_detected = poisoned_df

        print("***************** Unlearning the Backdoor *****************")
        # Use the detected poisoned samples to unlearn
        retain_loader = DataLoader(TargetDataset(tokenizer, args.max_len, clean_df_detected), batch_size=args.batch_size, shuffle=False)
        forget_loader = DataLoader(TargetDataset(tokenizer, args.max_len, poisoned_df_detected), batch_size=args.batch_size, shuffle=False)

        print(f"Unlearning Method:{args.unlearning_method}")
        if args.unlearning_method == "RT":
            """Retrain the model from scratch using the clean data as oracle baseline"""
            print("Defender: Retraining")
            initial_model = BertClassification(BertModel.from_pretrained(args.victim_model), label_num=num_class).to(device)
            for epoch in range(args.poisoning_epoch):
                print("\n-------------------")
                train_loss = train(initial_model, train_loader_clean, args, device)
                print(f"Epoch {epoch + 1}/{args.poisoning_epoch} | Train loss {train_loss}")

            print("-------------------------------------------------")
            print("---------------ReTrain Performance---------------")
            print("Clean Performance: Evaluation on Clean Test Set...")
            evaluate(initial_model, test_loader_clean, args.dataset, device)

            print("Poisoned Performance: Evaluation on Poisoned Test Set...")
            evaluate(initial_model, test_loader_poisoned, args.dataset, device)

        elif args.unlearning_method == "GA":
            """Use Gradient Ascent to unlearn the poisoned samples"""
            print("Defender: Gradient Ascent")
            for epoch in range(args.unlearning_epoch):
                print("\n-------------------")
                clean_loss, poison_loss = ga_train(target_model, retain_loader, forget_loader, args, device)
                print(f"Epoch {epoch + 1}/{args.unlearning_epoch} | Clean loss {clean_loss} | Poison loss {poison_loss}")

                if (epoch + 1) % 10 == 0:
                    print("-------------------------------------------------")
                    print("-----------Gradient Ascent Performance-----------")
                    print(f"Unlearning Epoch: {epoch + 1}")
                    print("Clean Performance: Evaluation on Clean Test Set...")
                    evaluate(target_model, test_loader_clean, args.dataset, device)

                    print("Poisoned Performance: Evaluation on Poisoned Test Set...")
                    evaluate(target_model, test_loader_poisoned, args.dataset, device)

        elif args.unlearning_method == "NPO":
            """Use NPO to unlearn the poisoned samples"""
            print("Defender: NPO")
            ref_model = copy.deepcopy(target_model)
            for epoch in range(args.unlearning_epoch):
                print("\n-------------------")
                clean_loss, poison_loss = npo_train(target_model, ref_model, retain_loader, forget_loader, args, device)
                print(f"Epoch {epoch + 1}/{args.unlearning_epoch} | Clean loss {clean_loss} | Poison loss {poison_loss}")

                if (epoch + 1) % 10 == 0:
                    print("-------------------------------------------------")
                    print("------------------NPO Performance----------------")
                    print(f"Unlearning Epoch: {epoch + 1}")
                    print("Clean Performance: Evaluation on Clean Test Set...")
                    evaluate(target_model, test_loader_clean, args.dataset, device)

                    print("Poisoned Performance: Evaluation on Poisoned Test Set...")
                    evaluate(target_model, test_loader_poisoned, args.dataset, device)

        elif args.unlearning_method == "RGA":
            """Use RGA to unlearn the poisoned samples"""
            print("Defender: RGA")
            logits_ref = compute_logits_ref(target_model, forget_loader, device)
            model_base = BertModel.from_pretrained(args.victim_model).to(device)

            for epoch in range(args.unlearning_epoch):
                print("\n-------------------")
                clean_loss, poison_loss = rga_train(target_model, model_base, logits_ref, retain_loader, forget_loader, args, device)
                print(f"Epoch {epoch + 1}/{args.unlearning_epoch} | Clean loss {clean_loss} | Poison loss {poison_loss}")

                if (epoch + 1) % 10 == 0:
                    print("-------------------------------------------------")
                    print("------------------RGA Performance----------------")
                    print(f"Unlearning Epoch: {epoch + 1}")
                    print("Clean Performance: Evaluation on Clean Test Set...")
                    evaluate(target_model, test_loader_clean, args.dataset, device)

                    print("Poisoned Performance: Evaluation on Poisoned Test Set...")
                    evaluate(target_model, test_loader_poisoned, args.dataset, device)

        else:
            raise ValueError(f"Unknown unlearning method: {args.unlearning_method}")
        print(f"================= Finished Seed: {current_seed} =================")

        gc.collect()
        torch.cuda.empty_cache()
