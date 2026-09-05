# Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization
This is the repository for [Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization](https://arxiv.org/abs/2505.16467).

Authors: Vera Neplenbroek, Arianna Bisazza, Raquel Fernández.

![Introduction](figure.png)

## Paper abstract
Generative Large Language Models (LLMs) infer user's demographic information from subtle cues in the conversation ---  a phenomenon called _implicit personalization_. Prior work has shown that such inferences can lead to lower quality
responses for users assumed to be from minority groups, even when no demographic information is explicitly provided. In this work, we systematically explore how LLMs respond to stereotypical cues using controlled synthetic conversations, by
analyzing the models' latent user representations through both model internals and generated answers to targeted user questions. Our findings reveal that LLMs do infer demographic attributes based on these stereotypical signals, which for a
number of groups even persists when the user explicitly identifies with a different demographic group. Finally, we show that this form of stereotype-driven implicit personalization can be effectively mitigated by intervening on the model's
internal representations using a trained linear probe to steer them toward the explicitly stated identity. Our results highlight the need for greater transparency and control in how LLMs represent user identity.

## Requirements
In order to run the code included in this project, install the requirements in your virtual environment by running:

```
pip install -r requirements.txt
```
This project was developed using Python 3.12.

This repository contains snippets of code from:
- https://github.com/yc015/TalkTuner-chatbot-llm-dashboard

## Using this repository
- `eval_conversations` contains the code to evaluate a model on all conversations for a demographic, including mitigation using probe weights wherever applicable.

  Example usage:
  ```
  python eval_conversations.py -m allenai/OLMo-2-1124-7B-Instruct -n 250 -bs 8 -d gender
  ```
- `train_probe.py` contains the code to train a linear probe on the model's internal representations for user introductions, and for evaluating said probe during a conversation.

  Example usage:
  ```
  python train_probe.py -m allenai/OLMo-2-1124-7B-Instruct --save_probe
  ```
- `mitigate.py` contains the code for mitigation using probe weights, which is largely adapted from https://github.com/yc015/TalkTuner-chatbot-llm-dashboard.
- `utils.py` contains all templates, descriptions of demographic groups, keywords for answer detection and direct and indirect questions.
- `stereotypes.csv` contains all stereotypical items we collected, as well as the set of neutral items.
- `conversations_250.json` contains all conversations we use in the paper, 250 per type.
- `visualizations.ipynb` contains the code for generating the visualizations, tables and statistical tests included in the paper.

## Citation
If you use the code in this repository, please cite the following paper:
```
@inproceedings{neplenbroek-etal-2025-reading,
    title = "Reading Between the Prompts: How Stereotypes Shape {LLM}{'}s Implicit Personalization",
    author = "Neplenbroek, Vera  and
      Bisazza, Arianna  and
      Fern{\'a}ndez, Raquel",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1029/",
    doi = "10.18653/v1/2025.emnlp-main.1029",
    pages = "20367--20400",
    ISBN = "979-8-89176-332-6",
    abstract = "Generative Large Language Models (LLMs) infer user{'}s demographic information from subtle cues in the conversation {---} a phenomenon called implicit personalization. Prior work has shown that such inferences can lead to lower quality responses for users assumed to be from minority groups, even when no demographic information is explicitly provided. In this work, we systematically explore how LLMs respond to stereotypical cues using controlled synthetic conversations, by analyzing the models' latent user representations through both model internals and generated answers to targeted user questions. Our findings reveal that LLMs do infer demographic attributes based on these stereotypical signals, which for a number of groups even persists when the user explicitly identifies with a different demographic group. Finally, we show that this form of stereotype-driven implicit personalization can be effectively mitigated by intervening on the model{'}s internal representations using a trained linear probe to steer them toward the explicitly stated identity. Our results highlight the need for greater transparency and control in how LLMs represent user identity."
}
```
