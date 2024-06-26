"""
Infer trained model.
"""
import argparse
import logging
import os
from pathlib import Path

import torch
from dataset_lifted import load_split_dataset
from eval import evaluate_sym_trans
from s2s_hf_transformers import HF_MODELS, T5_PREFIX
from s2s_pt_transformer import (
    DIM_FFN_HID,
    EMBED_SIZE,
    NHEAD,
    NUM_DECODER_LAYERS,
    NUM_ENCODER_LAYERS,
    Seq2SeqTransformer,
)
from s2s_pt_transformer import (
    construct_dataset_meta as pt_transformer_construct_dataset_meta,
)
from s2s_pt_transformer import translate as pt_transformer_translate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import count_params


S2S_MODELS = HF_MODELS.extend(["pt_transformer"])


class Seq2Seq:
    def __init__(self, model_dpath, model_name, **kwargs):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "t5" in model_name or "bart" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dpath)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dpath)
        elif model_name == "pt_transformer":
            self.model = Seq2SeqTransformer(
                kwargs["src_vocab_sz"],
                kwargs["tar_vocab_sz"],
                NUM_ENCODER_LAYERS,
                NUM_DECODER_LAYERS,
                EMBED_SIZE,
                NHEAD,
                DIM_FFN_HID,
            )
            self.model_translate = pt_transformer_translate
            self.vocab_transform = vocab_transform
            self.text_transform = text_transform
            self.model.load_state_dict(torch.load(kwargs["fpath_load"]))
        else:
            raise ValueError(f"ERROR: unrecognized model: {model_name}")

    def translate(self, queries):
        if "t5" in self.model_name or "bart" in self.model_name:
            inputs = [f"{T5_PREFIX}{query}" for query in queries]  # add prefix
            inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(
                self.device
            )
            output_tokens = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,
                max_new_tokens=256,
            )
            ltls = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        elif self.model_name == "pt_transformer":
            ltls = [
                self.model_translate(
                    self.model, self.vocab_transform, self.text_transform, queries[0]
                )
            ]
        else:
            raise ValueError(f"ERROR: unrecognized model, {self.model_name}")
        return ltls

    def parameters(self):
        return self.model.parameters()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_fpath",
        type=str,
        default="data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_0.pkl",
        help="complete file path or prefix of file paths to train test split dataset.",
    )
    parser.add_argument(
        "--model_dpath",
        type=str,
        default=None,
        help="directory to save model checkpoints.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="t5-base",
        choices=S2S_MODELS,
        help="name of supervised seq2seq model.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="checkpoint for inference."
    )
    args = parser.parse_args()

    ckpt_dname = (
        f"checkpoint-{args.checkpoint}" if args.checkpoint else "checkpoint-best"
    )

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            logging.FileHandler(
                f"results/s2s_{args.model}_{Path(args.data_fpath).stem}.log", mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    if (
        "pkl" in args.data_fpath
    ):  # complete file path, e.g. data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_0.pkl
        data_fpaths = [args.data_fpath]
    else:  # prefix of file paths, e.g. data/holdout_split_batch12_perm/symbolic_batch12_perm_utt
        data_dpath = os.path.dirname(args.data_fpath)
        fname_prefix = os.path.basename(args.data_fpath)
        data_fpaths = [
            os.path.join(data_dpath, fpath)
            for fpath in os.listdir(data_dpath)
            if fname_prefix in fpath
        ]

    for data_fpath in data_fpaths:
        logging.info(f"Inference dataset: {data_fpath}")
        # Load train, test data
        train_iter, train_meta, valid_iter, valid_meta = load_split_dataset(data_fpath)

        # Load trained model
        if (
            "t5" in args.model or "bart" in args.model
        ):  # pretrained T5/Bart from Hugging Face
            model_dpath = os.path.join(args.model_dpath, args.model, ckpt_dname)
            logging.info(f"Load model and checkpoint: {model_dpath}")
            s2s = Seq2Seq(model_dpath, args.model)
        elif (
            args.model == "pt_transformer"
        ):  # pretrained seq2seq transformer implemented in PyTorch
            (
                vocab_transform,
                text_transform,
                src_vocab_size,
                tar_vocab_size,
            ) = pt_transformer_construct_dataset_meta(train_iter)
            model_params = f"model/s2s_{args.model}_{Path(data_fpath).stem}.pth"
            s2s = Seq2Seq(
                args.model_dpath,
                args.model,
                vocab_transform=vocab_transform,
                text_transform=text_transform,
                src_vocab_sz=src_vocab_size,
                tar_vocab_sz=tar_vocab_size,
                fpath_load=model_params,
            )
        else:
            raise TypeError(f"ERROR: unrecognized model, {args.model}")
        logging.info(
            f"Number of trainable parameters in {args.model}: {count_params(s2s)}"
        )
        logging.info(f"Number of training samples: {len(train_iter)}")
        logging.info(f"Number of validation samples: {len(valid_iter)}\n")

        # Evaluate
        result_log_fpath = (
            f"results/s2s_{args.model}-{ckpt_dname}_{Path(data_fpath).stem}_log.csv"
        )
        analysis_fpath = "data/analysis_symbolic_batch12_perm.csv"
        acc_fpath = (
            f"results/s2s_{args.model}-{ckpt_dname}_{Path(data_fpath).stem}_acc.csv"
        )
        evaluate_sym_trans(s2s, data_fpath, result_log_fpath, analysis_fpath, acc_fpath)
