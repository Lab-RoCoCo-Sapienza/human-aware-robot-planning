import csv
import os
import pickle

import openai
from Lang2LTL.formula_sampler import ALL_PROPS
from Lang2LTL.get_embed import generate_embeds
from Lang2LTL.lang2ltl import (
    SHARED_DPATH,
    ground_res,
    ground_utterances,
    rer,
    translate_grounded_utts,
)
from Lang2LTL.s2s_hf_transformers import HF_MODELS
from Lang2LTL.utils import load_from_file
from utils import check_equivalence, prefix_to_infix_LTL


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sentence_2_formula(
    utt,
    obj2sem,
    rer_model,
    rer_engine,
    rer_prompt,
    model_fpath,
    sym_trans_model,
    convert_rule,
    props,
    target_formula=None,
    log_file=None,
):
    data_dpath = f"{SHARED_DPATH}/data"
    exp_name = "lang2ltl-api"
    embed_engine = "text-embedding-ada-002"
    ground_model = "gpt3"
    topk = 2
    update_embed = True
    keep_keys = None
    # keep_keys = ["description"]
    # keep_keys = ["key"]
    embed_model = "gpt3"
    """
    Extract referring expressions to use by a vision-language model to resolve propositions.
    :param utt: input utterance.
    :param rer_model: referring expression recognition module, e.g., "gpt3", "gpt4", "llama-7B".
    :param rer_engine: GPT engine for RER, e.g., "text-davinci-003", "gpt4".
    :param rer_prompt: prompt for GPT RER.
    :param model_fpath: pretrained model weights for symbolic translation.
    :param sym_trans_model: symbolic translation module, e.g., "t5-base", "gpt3_finetuned", "gpt3_pretrained".
    :param convert_rule: conversion rule from referring expressions to propositions, e.g., "lang2ltl", "cleanup".
    :param props: all available propositions.
    """
    res, utt2res = rer(rer_model, rer_engine, rer_prompt, [utt])
    # print("res: {}\tutt2res: {}")

    obj2embed, obj2embed_fpath = generate_embeds(
        embed_model,
        data_dpath,
        obj2sem,
        keep_keys=keep_keys,
        embed_engine=embed_engine,
        exp_name=exp_name,
        update_embed=update_embed,
    )
    print(
        f"Generated Database of Embeddings for:\n{obj2sem}\nsaved at:\n{obj2embed_fpath}\n"
    )

    with open(obj2embed_fpath, "rb") as f:
        content = pickle.load(f)
        print(content.keys())

    re2embed_dpath = os.path.join(data_dpath, "re_embeds")
    os.makedirs(re2embed_dpath, exist_ok=True)
    re2embed_fpath = os.path.join(
        re2embed_dpath, f"re2embed_{exp_name}_{embed_model}-{embed_engine}.pkl"
    )
    re2grounds = ground_res(
        res, re2embed_fpath, obj2embed_fpath, ground_model, embed_engine, topk
    )
    print(f"Groundings for REs:\n{re2grounds}\n")

    ground_utts, objs_per_utt = ground_utterances([utt], utt2res, re2grounds)
    print(
        f"Grounded Input Utterance:\n{ground_utts[0]}\ngroundings: {objs_per_utt[0]}\n"
    )

    if sym_trans_model == "gpt3_finetuned":
        translation_engine = "gpt3_finetuned_symbolic_batch12_perm_utt_0.2_42"
        translation_engine = load_from_file(
            os.path.join(model_fpath, "gpt3_models.pkl")
        )[translation_engine]
    elif sym_trans_model in HF_MODELS:
        sym_trans_model_path = os.path.join("Lang2LTL", sym_trans_model)
        model_fpath = os.path.join(sym_trans_model_path, model_fpath, "checkpoint-best")
        translation_engine = model_fpath
    else:
        raise ValueError(
            f"ERROR: unrecognized symbolic translation model: {sym_trans_model}"
        )

    (
        symbolic_utts,
        symbolic_ltls,
        output_ltls,
        placeholder_maps,
    ) = translate_grounded_utts(
        ground_utts,
        objs_per_utt,
        sym_trans_model,
        translation_engine,
        convert_rule,
        props,
    )
    print(
        f"Input utt: {utt}\n\nSymbolic utt: {symbolic_utts}\n\nSymbolic ltl: {symbolic_ltls}\n\nPlaceholder map: {placeholder_maps}\n\nOutput ltl: {output_ltls}"
    )

    predicted_formula, _ = prefix_to_infix_LTL(output_ltls[0])
    print("formula in infix format: ", predicted_formula)
    if target_formula is not None:
        correct = check_equivalence(predicted_formula, target_formula)
    else:
        correct = None

    if log_file is not None:
        with open(log_file, "a", newline="") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(
                [
                    utt.replace(";", ","),
                    str(placeholder_maps[0]).replace(";", ","),
                    symbolic_utts[0].replace(";", ","),
                    symbolic_ltls[0],
                    output_ltls[0],
                    target_formula,
                    correct,
                ]
            )

    return predicted_formula, correct


def test_on_a_set(dataset_csv, landmark_file, rer_prompt_file, output_csv):
    # prepare output file
    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        field = [
            "input utt",
            "placeholder map",
            "symbolic utt",
            "symbolic ltl",
            "output ltl",
            "target ltl",
            "correct",
        ]
        writer.writerow(field)
    correct = 0.0
    total = 0.0
    with open(dataset_csv, newline="") as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file, delimiter=";")

        # Iterate over each row in the CSV file
        for row in csv_reader:
            sentence = row[0]
            target_formula = row[1]
            finalLTL, correct = sentence_2_formula(
                sentence,
                landmark_file,
                rer_model="gpt3",
                rer_engine="gpt-3.5-turbo-instruct",
                rer_prompt=rer_prompt_file,
                model_fpath=f"composed_model_3000000/",
                sym_trans_model="t5-base",
                convert_rule="lang2ltl",
                props=ALL_PROPS,
                target_formula=target_formula,
                log_file=output_csv,
            )
            print(
                "--------------------------final LTL formula---------------------------"
            )
            print(finalLTL)
            if correct:
                print(
                    "The predicted formula is EQUIVALENT to the target:", target_formula
                )
                correct += 1
            else:
                print(
                    "The predicted formula is NOT equivalent to the target:",
                    target_formula,
                )
            total += 1
