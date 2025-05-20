import argparse
from collections import Counter
from copy import copy
from itertools import permutations
import json
import numpy as np
import os
import pandas as pd
import pickle
import re
import string
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from mitigate import get_layer_names, modified_model, modified_model_surpr
from train_probe import (
    get_introductions,
    eval_sample,
    get_chat_repr,
    get_surprisal,
)
from utils import (
    answer_detect_map,
    get_year,
    get_age,
    explicit_indicators,
    neutral_introductions,
    templates,
    token_values,
    indirect_questions,
    direct_questions,
    val_to_id,
)

import warnings


def detect_indirect_q_answer(answer_detect_map, question, answer, demographic):
    """Detects answer from indirect question using keywords"""
    token_answer = (
        answer.translate(
            str.maketrans("", "", "!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~")
        )
        .lower()
        .split()
    )
    demo_term = ""
    for token in answer_detect_map:
        if token in token_answer:
            if demo_term and demo_term != answer_detect_map[token]:
                demo_term = "mixed"
            else:
                if (
                    "pronouns" in question
                    and token == "them"
                    and token_answer[token_answer.index(token) - 1]
                    in ["use", "include"]
                ):
                    continue
                if (
                    token in ["low", "lower", "high", "higher"]
                    and token_answer.index(token) < (len(token_answer) - 1)
                    and token_answer[token_answer.index(token) + 1] != "income"
                ):
                    continue
                demo_term = answer_detect_map[token]
    if (
        demographic == "age"
        and not demo_term
        and ("year" in question or "born" in question)
    ):
        nums = [word[:4] for word in token_answer if word[:4].isnumeric()]
        if nums:
            demo_term = get_year(int(nums[0]))
    if demographic == "age" and not demo_term:
        nums = [
            word.split("s")[0]
            for word in token_answer
            if word.split("s")[0].isnumeric()
        ]
        if nums:
            demo_term = get_age(int(nums[0]))
    return demo_term


def detect_direct_q_answer(answer_detect_map, answer, demographic):
    """Detects answer from direct question using keywords"""
    answer = (
        answer.translate(
            str.maketrans("", "", "!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~")
        )
        .lower()
        .split()
    )
    demo_term = ""
    for token in answer_detect_map:
        if token in answer:
            if demo_term and demo_term != answer_detect_map[token]:
                demo_term = "mixed"
            else:
                demo_term = answer_detect_map[token]
    if demographic == "age" and not demo_term:
        nums = [
            word.split("-")[0]
            for word in answer
            if word.split("-")[0].isnumeric()
        ]
        if nums:
            demo_term = get_age(int(nums[0]))
    return demo_term


def get_conversations(
    demographic,
    value,
    stereotypes,
    templates,
    n=None,
    neutral=False,
):
    """Creates conversations by randomly selecting topics and items and slotting those into matching templates"""
    if neutral:
        neutral_items = stereotypes[stereotypes["dem_attribute"] == "neutral"][
            ["type", "value"]
        ].values
        np.random.shuffle(neutral_items)
        neutral_convos = [
            [
                np.random.choice(templates[item[0]]).format(item[1])
                for item in comb
            ]
            for comb in permutations(neutral_items, 6)
        ][:n]
        return neutral_convos
    else:
        stereo_convos = []
        stereo_items = stereotypes[
            stereotypes["dem_attribute"] == demographic
        ][stereotypes["dem_value"] == value][["type", "value"]].values
        if len(stereo_items):
            stereo_convos = [
                [
                    np.random.choice(templates[item[0]]).format(item[1])
                    for item in stereo_items[
                        np.random.choice(len(stereo_items), 6, replace=False)
                    ]
                ]
                for _ in range(n)
            ]
        return stereo_convos


def eval_convos(
    model,
    tokenizer,
    batch_size,
    device,
    convos,
    demographic,
    num_values,
    save_file,
    open_questions,
    indirect_questions,
    direct_question,
    answer_detect_map,
    val_to_id,
    n_layers,
    model_name,
    N,
    target=None,
):
    """Evaluate one type of conversations for one demographic at a time, including probe, surprisal, direct and indirect questions, and steering towards the target group if applicable"""
    probes = {
        n: pickle.load(
            open(
                save_file + f"_{demographic}_{n}.pkl",
                "rb",
            )
        )
        for n in range(n_layers)
    }
    modified_layer_names = get_layer_names(model.model, model_name)
    results = {
        i: {
            "probe": np.zeros((n_layers, num_values + 1)),
            "surprisal": np.zeros((len(convos), num_values)),
            "mod_surprisal": np.zeros((len(convos), num_values)),
            "direct_question": np.zeros(num_values + 2),
            "indirect_question": np.zeros((5, num_values + 2)),
            "mod_direct_question": np.zeros(num_values + 2),
            "mod_indirect_question": np.zeros((5, num_values + 2)),
        }
        for i in [0, 1, 3, 6]
    }
    return_convos = {i: {} for i in [0, 1, 3, 6]}
    current_convos = [[] for _ in range(len(convos))]
    user_turns = [
        [{"role": "user", "content": turn} for turn in convo]
        for convo in convos
    ]
    for i in range(7):
        print(i)
        for j, current_convo in enumerate(current_convos):
            current_convo.append(user_turns[j][i])
            if i in [0, 1, 3, 6]:
                repr = get_chat_repr(
                    current_convo[:],
                    demographic,
                    model.model,
                    tokenizer,
                    device,
                )
                demo_terms = eval_sample(repr, demographic, save_file)
                for k, demo_term in enumerate(demo_terms):
                    results[i]["probe"][k, val_to_id[demo_term]] += 1

                surprisal = get_surprisal(
                    current_convo[:],
                    demographic,
                    model.model,
                    tokenizer,
                    device,
                    values=token_values[demographic],
                    bow_token="OLMo" not in model_name,
                )
                for val in surprisal:
                    results[i]["surprisal"][j][val_to_id[val]] = surprisal[val]
                if target:
                    new_surprisal = modified_model_surpr(
                        model.model,
                        tokenizer,
                        probes,
                        modified_layer_names,
                        demographic,
                        target,
                        current_convo[:],
                        device,
                        token_values[demographic],
                        N,
                        bow_token="OLMo" not in model_name,
                    )
                    for val in new_surprisal:
                        results[i]["mod_surprisal"][j][val_to_id[val]] = (
                            new_surprisal[val]
                        )
        current_convos = [
            convo[0]["generated_text"]
            for convo in tqdm(
                model(
                    current_convos,
                    batch_size=batch_size,
                    do_sample=False,
                    max_new_tokens=100,
                ),
                total=len(current_convos),
            )
        ]

        if i in [0, 1, 3, 6]:
            return_convos[i]["indirect_question"] = {}
            return_convos[i]["mod_indirect_question"] = {}
            for k, indirect_question in enumerate(indirect_questions):
                question_convos = [
                    current_convo
                    + [{"role": "user", "content": indirect_question}]
                    for current_convo in current_convos
                ]
                model_answers = [
                    answer[0]["generated_text"][-1]["content"]
                    for answer in tqdm(
                        model(
                            question_convos,
                            batch_size=batch_size,
                            do_sample=False,
                            max_new_tokens=100,
                        ),
                        total=len(question_convos),
                    )
                ]
                return_convos[i]["indirect_question"][k] = model_answers
                demo_terms = list(
                    map(
                        lambda x: detect_indirect_q_answer(
                            answer_detect_map[demographic],
                            indirect_question,
                            x,
                            demographic,
                        ),
                        model_answers,
                    )
                )
                demo_term_counts = Counter(demo_terms)
                for demo_term in demo_term_counts:
                    results[i]["indirect_question"][
                        k, val_to_id[demo_term]
                    ] += demo_term_counts[demo_term]
                if target:
                    new_answers = modified_model(
                        model,
                        probes,
                        modified_layer_names,
                        demographic,
                        target,
                        batch_size,
                        question_convos,
                        N,
                    )
                    return_convos[i]["mod_indirect_question"][k] = new_answers
                    demo_terms = list(
                        map(
                            lambda x: detect_indirect_q_answer(
                                answer_detect_map[demographic],
                                indirect_question,
                                x,
                                demographic,
                            ),
                            new_answers,
                        )
                    )
                    demo_term_counts = Counter(demo_terms)
                    for demo_term in demo_term_counts:
                        results[i]["mod_indirect_question"][
                            k, val_to_id[demo_term]
                        ] += demo_term_counts[demo_term]
            question_convos = [
                current_convo + [{"role": "user", "content": direct_question}]
                for current_convo in current_convos
            ]
            model_answers = [
                answer[0]["generated_text"][-1]["content"]
                for answer in tqdm(
                    model(
                        question_convos,
                        batch_size=batch_size,
                        do_sample=False,
                        max_new_tokens=100,
                    ),
                    total=len(question_convos),
                )
            ]
            return_convos[i]["direct_question"] = model_answers
            demo_terms = list(
                map(
                    lambda x: detect_direct_q_answer(
                        answer_detect_map[demographic], x, demographic
                    ),
                    model_answers,
                )
            )
            demo_term_counts = Counter(demo_terms)
            for demo_term in demo_term_counts:
                results[i]["direct_question"][
                    val_to_id[demo_term]
                ] += demo_term_counts[demo_term]
            if target:
                new_answers = modified_model(
                    model,
                    probes,
                    modified_layer_names,
                    demographic,
                    target,
                    batch_size,
                    question_convos,
                    N,
                )
                return_convos[i]["mod_direct_question"] = new_answers
                demo_terms = list(
                    map(
                        lambda x: detect_direct_q_answer(
                            answer_detect_map[demographic], x, demographic
                        ),
                        new_answers,
                    )
                )
                demo_term_counts = Counter(demo_terms)
                for demo_term in demo_term_counts:
                    results[i]["mod_direct_question"][
                        val_to_id[demo_term]
                    ] += demo_term_counts[demo_term]
    return_convos["conversation"] = current_convos
    return results, return_convos


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch._dynamo.config.suppress_errors = True
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to evaluate",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Number of samples",
    )
    parser.add_argument(
        "-d",
        "--demo",
        type=str,
        default="gender",
        help="Demographic group to evaluate",
    )
    parser.add_argument(
        "-rd",
        "--results_dir",
        type=str,
        default="",
        help="Directory for storing results",
    )
    args = parser.parse_args()
    np.random.seed(42)
    stereotypes = pd.read_csv("stereotypes.csv").drop(columns=["source"])
    demo_introductions = get_introductions(
        templates["introductions"], explicit_indicators
    )
    natural_introductions = get_introductions(
        natural_intros, {"gender": explicit_indicators["gender"]}
    )
    content = {}
    if os.path.isfile(f"conversations_{args.n}.json"):
        with open(
            f"conversations_{args.n}.json",
            "r",
        ) as infile:
            conversations = json.load(infile)
    else:
        content["neutral"] = get_conversations(
            None, None, stereotypes, templates, n=args.n, neutral=True
        )
        for demographic in explicit_indicators:
            content[demographic] = {}
            for value in explicit_indicators[demographic]:
                content[demographic][value] = get_conversations(
                    demographic,
                    value,
                    stereotypes,
                    templates,
                    n=args.n,
                )
        conversations = {}
        neutral_intros = [
            str(np.random.choice(neutral_introductions)) for _ in range(args.n)
        ]
        for demographic in explicit_indicators:
            conversations[demographic] = {}
            conversations[demographic]["neutral_none"] = [
                [neutral_intros[i]] + content["neutral"][i]
                for i in range(len(neutral_intros))
            ]
            for value in explicit_indicators[demographic]:
                conversations[demographic][value] = {}
                demo_intros = [
                    str(
                        np.random.choice(
                            demo_introductions[demographic][value]
                        )
                    )
                    for _ in range(args.n)
                ]
                conversations[demographic][value]["neutral_demo"] = [
                    [demo_intros[i]] + content["neutral"][i]
                    for i in range(len(demo_intros))
                ]
                anti_values = [
                    val
                    for val in stereotypes[
                        stereotypes["dem_attribute"] == demographic
                    ]["dem_value"].unique()
                    if val != value
                ]
                conversations[demographic][value]["anti_demo"] = {
                    val: [
                        [demo_intros[i]] + content[demographic][val][i]
                        for i in range(len(demo_intros))
                    ]
                    for val in anti_values
                }
                if len(content[demographic][value]):
                    conversations[demographic][value]["stereo_none"] = [
                        [neutral_intros[i]] + content[demographic][value][i]
                        for i in range(len(neutral_intros))
                    ]
                if demographic == "gender":
                    natural_intros = [
                        str(
                            np.random.choice(
                                natural_introductions[demographic][value]
                            )
                        )
                        for _ in range(args.n)
                    ]
                    conversations[demographic][value]["neutral_natural"] = [
                        [natural_intros[i]] + content["neutral"][0]
                        for i in range(len(natural_intros))
                    ]
                    conversations[demographic][value]["anti_natural"] = {
                        val: [
                            [natural_intros[i]] + content[demographic][val][0]
                            for i in range(len(natural_intros))
                        ]
                        for val in anti_values
                    }
                    if len(content[demographic][value]):
                        conversations[demographic][value]["stereo_natural"] = [
                            [natural_intros[i]]
                            + content[demographic][value][0]
                            for i in range(len(natural_intros))
                        ]

    with open(
        f"conversations_{args.n}.json",
        "w",
    ) as outfile:
        json.dump(conversations, outfile)

    if os.path.isfile(
        f"{args.rd}/results_{args.demo}_{args.n}_{args.model.split('/')[1]}.pkl"
    ):
        with open(
            f"{args.rd}/results_{args.demo}_{args.n}_{args.model.split('/')[1]}.pkl",
            "rb",
        ) as infile:
            results = pickle.load(infile)
        with open(
            f"{args.rd}/convos_{args.demo}_{args.n}_{args.model.split('/')[1]}.pkl",
            "rb",
        ) as infile:
            final_convos = pickle.load(infile)
    else:
        results = {}
        final_convos = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if "gemma" in args.model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if not model.tokenizer.pad_token_id:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    if "gemma-2-9b" in args.model:
        n_layers = 43
        N = 200
    elif "Llama-3.1-8B" in args.model:
        n_layers = 33
        N = 1
    elif "OLMo-2" in args.model:
        n_layers = 33
        N = 2

    if "neutral_none" not in results:
        curr_results, curr_convos = eval_convos(
            model,
            tokenizer,
            args.batch_size,
            device,
            conversations[args.demo]["neutral_none"],
            args.demo,
            len(explicit_indicators[args.demo]),
            f"{args.rd}/{args.model.split('/')[1]}_probe_",
            open_questions,
            indirect_questions[args.demo],
            direct_questions[args.demo],
            answer_detect_map,
            val_to_id,
            n_layers,
            args.model,
            N,
            target=None,
        )
        results["neutral_none"] = curr_results
        final_convos["neutral_none"] = curr_convos
        with open(
            f"{args.rd}/results_{args.demo}_{args.n}_{args.model.split('/')[1]}.pkl",
            "wb",
        ) as outfile:
            pickle.dump(results, outfile)
        with open(
            f"{args.rd}/convos_{args.demo}_{args.n}_{args.model.split('/')[1]}.pkl",
            "wb",
        ) as outfile:
            pickle.dump(final_convos, outfile)
    for value in tqdm(explicit_indicators[args.demo]):
        if value not in results:
            results[value] = {}
            final_convos[value] = {}
        for convo_type in tqdm(conversations[args.demo][value]):
            if "anti" in convo_type:
                if convo_type not in results[value]:
                    results[value][convo_type] = {}
                    final_convos[value][convo_type] = {}
                for anti_val in conversations[args.demo][value][convo_type]:
                    if anti_val not in results[value][convo_type]:
                        print(demographic, value, convo_type, anti_val)
                        curr_results, curr_convos = eval_convos(
                            model,
                            tokenizer,
                            args.batch_size,
                            device,
                            conversations[args.demo][value][convo_type][
                                anti_val
                            ],
                            args.demo,
                            len(explicit_indicators[args.demo]),
                            f"{args.rd}/{args.model.split('/')[1]}_probe_",
                            open_questions,
                            indirect_questions[args.demo],
                            direct_questions[args.demo],
                            answer_detect_map,
                            val_to_id,
                            n_layers,
                            args.model,
                            N,
                            target=value,
                        )
                        results[value][convo_type][anti_val] = curr_results
                        final_convos[value][convo_type][anti_val] = curr_convos
                        with open(
                            f"{args.rd}/results_{args.demo}_{args.n}_{args.model.split('/')[1]}.pkl",
                            "wb",
                        ) as outfile:
                            pickle.dump(results, outfile)
                        with open(
                            f"{args.rd}/convos_{args.demo}_{args.n}_{args.model.split('/')[1]}.pkl",
                            "wb",
                        ) as outfile:
                            pickle.dump(final_convos, outfile)

            else:
                if convo_type not in results[value]:
                    print(args.demo, value, convo_type)
                    if "stereo" in convo_type:
                        target = "neutral"
                    else:
                        target = None
                    curr_results, curr_convos = eval_convos(
                        model,
                        tokenizer,
                        args.batch_size,
                        device,
                        conversations[args.demo][value][convo_type],
                        args.demo,
                        len(explicit_indicators[args.demo]),
                        f"{args.rd}/{args.model.split('/')[1]}_probe_",
                        open_questions,
                        indirect_questions[args.demo],
                        direct_questions[args.demo],
                        answer_detect_map,
                        val_to_id,
                        n_layers,
                        args.model,
                        N,
                        target=target,
                    )
                    results[value][convo_type] = curr_results
                    final_convos[value][convo_type] = curr_convos
                    with open(
                        f"{args.rd}/results_{args.demo}_{args.n}_{args.model.split('/')[1]}.pkl",
                        "wb",
                    ) as outfile:
                        pickle.dump(results, outfile)
                    with open(
                        f"{args.rd}/convos_{args.demo}_{args.n}_{args.model.split('/')[1]}.pkl",
                        "wb",
                    ) as outfile:
                        pickle.dump(final_convos, outfile)
