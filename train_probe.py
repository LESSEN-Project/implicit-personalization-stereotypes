import argparse
import itertools
import json
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from utils import (
    get_age,
    neutral_introductions,
    templates,
    explicit_indicators,
)
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_introductions(
    introductions,
    explicit_indicators,
):
    """Slot explicit indicators of the users demographic group into templates to create introductions"""
    filled_introductions = {}
    for demographic in explicit_indicators:
        filled_introductions[demographic] = {}
        for value in explicit_indicators[demographic]:
            if type(introductions) == dict:
                intros = introductions[demographic][value]
            else:
                intros = introductions
            filled_introductions[demographic][value] = []
            indicators = explicit_indicators[demographic][value]
            filled_introductions[demographic][value] += list(
                set(
                    [
                        i.format(indicator)
                        for indicator in indicators
                        for i in intros
                    ]
                )
            )
    return filled_introductions


def get_tokenized_chat(
    chat,
    demographic,
    tokenizer,
):
    """Apply chat template, add special sentence for probe/surprisal and tokenize chat"""
    chat = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=False,
    )
    chat += f" I think the {demographic} of this user is "
    tokenized_chat = tokenizer.encode(chat, return_tensors="pt")
    return tokenized_chat


def get_repr(
    introductions,
    model,
    tokenizer,
):
    """Get representations for user introductions for probe training"""
    representations = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for demographic in tqdm(introductions):
        representations[demographic] = {}
        for value in tqdm(introductions[demographic]):
            representations[demographic][value] = []
            for intro in introductions[demographic][value]:
                chat = [
                    {"role": "user", "content": intro},
                ]
                tokenized_chat = get_tokenized_chat(
                    chat,
                    demographic,
                    tokenizer,
                ).to(device)
                outputs = model(
                    tokenized_chat,
                    output_hidden_states=True,
                    max_new_tokens=1,
                    return_dict=True,
                )["hidden_states"]
                representations[demographic][value].append(
                    [
                        o[-1, -1, :].detach().cpu().clone().to(torch.float)
                        for o in outputs
                    ]
                )
    n_layers = len(outputs)
    return representations, n_layers


def get_surprisal(
    chat,
    demographic,
    model,
    tokenizer,
    device,
    values,
    bow_token=True,
):
    """Gets minimum suprisal value across all descriptors for each demographic group"""
    tokenized_chat = get_tokenized_chat(
        chat,
        demographic,
        tokenizer,
    ).to(device)
    outputs = model.generate(
        tokenized_chat,
        max_new_tokens=1,
        output_logits=True,
        return_dict_in_generate=True,
    )["logits"][0]
    neg_log_prob = -torch.log_softmax(outputs[-1, :], dim=-1)
    surprisal = {
        val: min(
            [
                neg_log_prob[tokenizer.encode(x)[int(bow_token)]]
                for x in values[val]
            ]
        )
        for val in values
    }
    return surprisal


def get_chat_repr(
    chat,
    demographic,
    model,
    tokenizer,
    device,
):
    """Obtains representations from conversation for inference with probe"""
    tokenized_chat = get_tokenized_chat(
        chat,
        demographic,
        tokenizer,
    ).to(device)
    outputs = model.generate(
        tokenized_chat,
        output_hidden_states=True,
        max_new_tokens=1,
        return_dict_in_generate=True,
    )["hidden_states"][0]
    return [
        o[-1, -1, :].detach().cpu().clone().to(torch.float) for o in outputs
    ]


def train_probe(repr, n_layers, results_file, save=False, save_file=""):
    """Trains linear probe on representations from user introductions"""
    results = {}
    for demographic in tqdm(repr):
        results[demographic] = []
        for l in tqdm(range(n_layers)):
            X = [
                rep[l]
                for value in repr[demographic]
                for rep in repr[demographic][value]
            ]
            y = list(
                itertools.chain.from_iterable(
                    [
                        [value] * len(repr[demographic][value])
                        for value in repr[demographic]
                    ]
                )
            )
            clf = LogisticRegression(random_state=42)
            if save:
                clf = clf.fit(X, y)
                with open(
                    save_file + f"_{demographic}_{l}.pkl", "wb"
                ) as outfile:
                    pickle.dump(clf, outfile)
            else:
                scores = cross_val_score(clf, X, y, cv=5)
                results[demographic].append(scores)
    if not save:
        with open(
            results_file,
            "wb",
        ) as outfile:
            pickle.dump(results, outfile)


def eval_sample(repr, demographic, save_file):
    """Evaluates probe on specific representation"""
    results = []
    for l in range(len(repr)):
        with open(save_file + f"_{demographic}_{l}.pkl", "rb") as infile:
            clf = pickle.load(infile)
        results.append(str(clf.predict(repr[l].unsqueeze(0))[0]))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to evaluate",
    )
    parser.add_argument(
        "-rd",
        "--results_dir",
        type=str,
        default="",
        help="Directory for storing results",
    )
    parser.add_argument(
        "--save_probe", action="store_true", help="Save trained probe"
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).to(device)

    introductions = get_introductions(
        templates["introductions"],
        explicit_indicators,
    )
    for demographic in introductions:
        introductions[demographic]["neutral"] = neutral_introductions

    repr, n_layers = get_repr(
        introductions,
        model,
        tokenizer,
    )
    with open(
        f"{args.rd}/{args.model.split('/')[1]}_repr.pkl",
        "wb",
    ) as outfile:
        pickle.dump((repr, n_layers), outfile)
    train_probe(
        repr,
        n_layers,
        f"{args.rd}/{args.model.split('/')[1]}_probe_results.pkl",
        save=args.save_probe,
        save_file=f"{args.rd}/{args.model.split('/')[1]}_probe",
    )
