# Pinpoint, Not Criticize
This repository is dedicated to reproducing "Pinpoint, Not Criticize: Refining Large Language Models via Fine-Grained Actionable Feedback" ([Arxiv Link](https://arxiv.org/pdf/2311.09336.pdf)) with open-source LLMs.

<p align="center">
  <img src="data/main.png" width="400" class="center">
</p>

## Getting Started
You can set up environments for FITO and COMET (used for MT evaluation) using Conda as follows:
```
git clone https://github.com/xu1998hz/reproduce_pinpoint.git
cd reproduce_pinpoint
conda env create -f environment.yml
conda env create -f comet.yml
```
## How to Use
### Baseline LLM Performance
To evaluate the performance of baseline LLMs (currently supporting Llama 2 and Mistral) in Machine Translation, ASQA and summarization tasks, run the following bash scripts. The results are also available in the `out` directory.
```
cd run
./mt_run.sh
./qa_run.sh
./summ_run.sh
```
### Feedback Model
To fine-tune the Llama-7b model with fine-grained text feedback, run the following script. The checkpoint will be available for download soon.
```
cd run
./ft_run.sh
```
### Inference

### Citation

```
@inproceedings{xu-etal-2024-llmrefine,
    title = "{LLMR}efine: Pinpointing and Refining Large Language Models via Fine-Grained Actionable Feedback",
    author = "Xu, Wenda  and
      Deutsch, Daniel  and
      Finkelstein, Mara  and
      Juraska, Juraj  and
      Zhang, Biao  and
      Liu, Zhongtao  and
      Wang, William Yang  and
      Li, Lei  and
      Freitag, Markus",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.92/",
    doi = "10.18653/v1/2024.findings-naacl.92",
    pages = "1429--1445",
    abstract = "Recent large language models (LLM) areleveraging human feedback to improve theirgeneration quality. However, human feedbackis costly to obtain, especially during inference.In this work, we propose LLMRefine, aninference time optimization method to refineLLM`s output. The core idea is to usea learned fine-grained feedback model topinpoint defects and guide LLM to refinethem iteratively. Using original LLM as aproposal of edits, LLMRefine searches fordefect-less text via simulated annealing, tradingoff the exploration and exploitation. Weconduct experiments on three text generationtasks, including machine translation, long-form question answering (QA), and topicalsummarization. LLMRefine consistentlyoutperforms all baseline approaches, achievingimprovements up to 1.7 MetricX points ontranslation tasks, 8.1 ROUGE-L on ASQA, 2.2ROUGE-L on topical summarization."
}
```
