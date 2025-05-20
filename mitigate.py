from baukit import TraceDict
from utils import probe_targets
import torch
from torch import nn
from train_probe import get_tokenized_chat
from tqdm import tqdm

# This file was adapted from https://github.com/yc015/TalkTuner-chatbot-llm-dashboard


def get_layer_names(model, model_name):
    """Get layer names from model"""
    which_layers = []
    if "gemma" in model_name:
        from_idx = 30
        to_idx = 40
    else:
        from_idx = 20
        to_idx = 30
    for name, module in model.named_modules():
        if name != "" and name[-1].isdigit():
            layer_num = name[
                name.rfind("model.layers.") + len("model.layers.") :
            ]
            if from_idx <= int(layer_num) < to_idx:
                which_layers.append(name)
    return which_layers


def optimize_one_inter_rep(
    inter_rep, layer_name, target, probe, mult, normalized=False
):
    """Add probe weights to model's internal representations"""
    global first_time
    tensor = (inter_rep.clone()).to("cuda").requires_grad_(True)
    rep_f = lambda: tensor
    probe_weights = torch.from_numpy(probe.coef_[target]).to("cuda")

    if normalized:
        cur_input_tensor = (
            rep_f() + probe_weights * mult * 100 / rep_f().norm()
        )
    else:
        cur_input_tensor = rep_f() + probe_weights * mult
    return cur_input_tensor.clone()


def edit_inter_rep_multi_layers(output, layer_name):
    """Apply probe weight adding to all relevant layers of the model"""
    layer_num = int(
        layer_name[layer_name.rfind("model.layers.") + len("model.layers.") :]
    )
    probe = probes_dict[layer_num]
    cloned_inter_rep = (
        output[0][:, -1].unsqueeze(0).detach().clone().to(torch.float)
    )
    with torch.enable_grad():
        cloned_inter_rep = optimize_one_inter_rep(
            cloned_inter_rep,
            layer_name,
            cf_target,
            probe,
            mult=mult,
            normalized=False,
        )
    output[0][:, -1] = cloned_inter_rep.to(torch.float16)
    return output


def modified_model(
    model,
    probes,
    modified_layer_names,
    demographic,
    value,
    batch_size,
    question_convos,
    N,
):
    """Apply steering towards demographic group to model and generate response"""
    global probes_dict
    probes_dict = probes
    global mult
    mult = N
    global cf_target
    cf_target = probe_targets[demographic][value]
    with TraceDict(
        model.model,
        modified_layer_names,
        edit_output=edit_inter_rep_multi_layers,
    ) as ret:
        model_answer = [
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
    return model_answer


def modified_model_surpr(
    model,
    tokenizer,
    probes,
    modified_layer_names,
    demographic,
    value,
    chat,
    device,
    values,
    N,
    bow_token=True,
):
    """Apply steering towards demographic group to model and obtain surprisal values"""
    global probes_dict
    probes_dict = probes
    global mult
    mult = N
    global cf_target
    cf_target = probe_targets[demographic][value]
    tokenized_chat = get_tokenized_chat(
        chat,
        demographic,
        tokenizer,
    ).to(device)
    with TraceDict(
        model,
        modified_layer_names,
        edit_output=edit_inter_rep_multi_layers,
    ) as ret:
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
