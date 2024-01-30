# reproduce_pinpoint
This is repo to reproduce Pinpoint, Not Criticize: Refining Large Language Models via Fine-Grained Actionable Feedback (https://arxiv.org/abs/2311.09336).

ASQA contains two references, denoted as ref1 and ref2.

Summ contains multiple references, denoted as refs, each can be separated by "******"
## Set up
We provide conda environments for FITO and comet (for machine translation eval).
```
conda env create -f environment.yml
conda env create -f comet.yml
```
## Usage
### Baseline LLM Performance
To test the baseline open-source LLM (currently Llama 2 and Mistral supported) capability on machine translation, long-form QA and summerization. simply run the following bash file. We also provide given results under `out`.
```
cd run
./mt_run.sh
./qa_run.sh
./summ_run.sh
```
### Finetune Llama

