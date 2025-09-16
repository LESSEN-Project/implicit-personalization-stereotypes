import argparse
import torch
import pickle
from transformers import AutoModelForCausalLM
from mitigate import get_layer_names
from utils import probe_targets

model_to_name = {
    "google/gemma-2-9b-it": "gemma",
    "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "allenai/OLMo-2-1124-7B-Instruct": "olmo",
}
model_to_n = {
    "google/gemma-2-9b-it": 200,
    "meta-llama/Llama-3.1-8B-Instruct": 1,
    "allenai/OLMo-2-1124-7B-Instruct": 2,
}
model_to_layers = {
    "google/gemma-2-9b-it": 43,
    "meta-llama/Llama-3.1-8B-Instruct": 33,
    "allenai/OLMo-2-1124-7B-Instruct": 33,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd",
        "--results_dir",
        type=str,
        default="",
        help="Directory for storing results",
    )
    args = parser.parse_args()
    for model_name in [
        "google/gemma-2-9b-it",
        "meta-llama/Llama-3.1-8B-Instruct",
        "allenai/OLMo-2-1124-7B-Instruct",
    ]:
        if "gemma" in model_name:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        for demographic, value in [
            ("age", "adolescent"),
            ("age", "adult"),
            ("age", "child"),
            ("age", "neutral"),
            ("age", "older adult"),
            ("gender", "female"),
            ("gender", "male"),
            ("gender", "neutral"),
            ("gender", "non-binary"),
            ("race", "asian"),
            ("race", "black"),
            ("race", "hispanic"),
            ("race", "neutral"),
            ("race", "white"),
            ("socio-economic status", "high"),
            ("socio-economic status", "low"),
            ("socio-economic status", "neutral"),
        ]:
            print(model_name, demographic, value)
            probes_dict = {
                n: pickle.load(
                    open(
                        f"{args.rd}/{model_name.split('/')[1]}_probe__{demographic}_{n}.pkl",
                        "rb",
                    )
                )
                for n in range(model_to_layers[model_name])
            }
            layers = get_layer_names(model, model_to_name[model_name])
            steer_config = {
                ".".join(layer.split(".")[1:]): {
                    "steering_vector": torch.from_numpy(
                        probes_dict[
                            int(
                                layer[
                                    layer.rfind("model.layers.")
                                    + len("model.layers.") :
                                ]
                            )
                        ].coef_[probe_targets[demographic][value]]
                    ),
                    "steering_coefficient": model_to_n[model_name],
                    "action": "add",
                }
                for layer in layers
            }
            torch.save(
                steer_config,
                f"{args.rd}/{model_to_name[model_name]}_{demographic}_{value}_config.pt",
            )
