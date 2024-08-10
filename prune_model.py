import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def apply_svd_to_linear(layer, rank):
    weight = layer.weight.data
    U, S, V = torch.svd(weight)
    U = U[:, :rank]
    S = S[:rank]
    V = V[:, :rank]
    new_weight = torch.mm(U, torch.mm(torch.diag(S), V.t()))
    layer.weight.data = new_weight

def prune_model(model, rank):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            apply_svd_to_linear(module, rank)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the pruned model')
    parser.add_argument('--pruning_criterion', type=str, default='ppl', help='Pruning criterion')
    parser.add_argument('--rank', type=int, default=128, help='Rank for SVD')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    prune_model(model, args.rank)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()

