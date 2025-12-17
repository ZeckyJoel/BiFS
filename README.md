# BiFS: Bidirectional Focal Selection for Knowledge-Based Visual Question Answering

## 📖 Abstract

While MLLM-based knowledge acquisition has become prevalent in **Knowledge-Based Visual Question Answering** (KB-VQA), such methods inevitably introduce noise due to inadequate instruction following or insufficient prompt engineering. We propose a **"focus before reasoning"** framework that pre-filters question-irrelevant visual objects through a bidirectional selection mechanism. Unlike existing approaches, we deploy both a **Selector** agent to identify relevant objects and an **Excluder** agent to explicitly filter out irrelevant ones, with a** Judger** agent resolving conflicts to produce the final focal context with interpretable rationales. These rationales then guide the **Reasoner**agent to produce knowledge-enhanced reasoning chains. We further leverage these reasoning chains as auxiliary supervision losses and design a focal-aware module to integrate focal context as knowledge into LLMs, enhancing reasoning performance. Experiments show that our method achieves state-of-the-art performance with **70.5%** on OK-VQA and **74.3%** on A-OKVQA, which are standard benchmarks in the KBVQA field, surpassing previous best methods by **2.3%** and **1.9%** respectively.

## 🧠 Framework Overview

<img src=".\resources\framework.jpg" title="" alt="" width="659">

## 📊 Main Results

| Our Method              | Base Model        | OK-VQA   | A-OKVQA  |
| ----------------------- | ----------------- | -------- | -------- |
| **BiFS**                | InstructBLIP (7B) | 66.2     | 70.3     |
| **BiFS**                | LLaVA-1.5 (7B)    | **70.5** | **74.3** |
| w/o *Excluder & Judger* | LLaVA-1.5 (7B)    | 67.5     | 71.8     |
| w/o *Rationales*        | LLaVA-1.5 (7B)    | 68.2     | 72.5     |
| w/o *Supervision*       | LLaVA-1.5 (7B)    | 68.9     | 72.1     |
| w/o *FocalAware*        | LLaVA-1.5 (7B)    | 69.5     | 73.0     |

## 💾 Data Download

To run the BiFS framework, you need to download the following datasets:

- ​**​OK-VQA Dataset​**​: Available at [https://okvqa.allenai.org/](https://okvqa.allenai.org/)
- ​**​A-OKVQA Dataset​**​: Available at https://github.com/allenai/aokvqa

Please follow the respective websites' instructions for downloading and organizing these datasets.

## 🛠️ Installation

```
git clone [this repository]
cd BiFS
python -m venv bifs_env
pip install -r requirements.txt
```

## 🏃Run Code

### Step 1: Bidirectional Focal Objects Selection

Execute the following scripts in order:

```
python BiFS/pipelines/caption.py
python BiFS/pipelines/selector.py
python BiFS/pipelines/excluder.py
python BiFS/pipelines/process_conflict.py
```

This step generates initial captions, selects relevant objects through the Selector agent, identifies irrelevant objects via the Excluder agent, and processes any conflicts between their decisions.

### Step 2: Focal Context and Knowledge Acquisition

Run these scripts to resolve conflicts and acquire focal context:

```
python BiFS/pipelines/judger.py
python BiFS/pipelines/get_focal_context.py
python BiFS/pipelines/reasoner.py
```

The Judger agent resolves conflicts between Selector and Excluder decisions, and the focal context is extracted based on the finalized focal objects.

### Step 3: Multitask Supervised Fine-tuning with Visual Enhancement

Finally, run the fine-tuning scripts for each dataset:

```
CUDA_VISIBLE_DEVICES=0 python finetune_aokvqa.py
CUDA_VISIBLE_DEVICES=0 python finetune_okvqa.py
```

This step performs multi-task fine-tuning with visual enhancement, jointly optimizing answer generation and knowledge reasoning tasks.

## 🔍Case Studies

Below are qualitative case studies illustrating how our method reasons about visual questions. 
The figures show:

- **Selectors**: Entities and rationales that directly contribute to answering the question.  
- **Excluders**: Entities and rationales that are present but irrelevant to the question.  
- **Judger**: Processes the conflict set to determine whether an entity is the focal object. 
- **Reasoner**: Synthesizes the selected focal context and rationales into a coherent knowledge reasoning chain.




