# BiFS: Bidirectional Focal Selection for Knowledge-Based Visual Question Answering

-----

## 📖 Abstract

Knowledge-Based Visual Question Answering (KB-VQA) is a multimodal task where models answer visual questions using external knowledge beyond image content. Although Multimodal Large Language Models (MLLMs) generate rich knowledge for KB-VQA, they often introduce noise—irrelevant information that hinders reasoning. To mitigate this, we proposed the ​**​Bidirectional Focal Selection (BiFS)​**​ framework, introducing a *"focus before reasoning"* paradigm that identifies and uses the most relevant visual information before knowledge generation. Our key innovation is a bidirectional selection mechanism where ​**​Selector​**​ and ​**​Excluder​**​ agents filter relevant and irrelevant objects, providing rationales for their choices. A ​**​Judger​**​ agent resolves conflicts between their decisions to reliably identify focal objects. We design a focal-aware module that enhances focal regions in visual features. Through multi-task learning with answer generation and knowledge reasoning, BiFS learns to produce accurate, grounded responses. Experiments show that BiFS reaches ​**​70.5%​**​ on OK-VQA and ​**​74.3%​**​ on A-OKVQA, improving over state-of-the-art methods by ​**​2.3%​**​ and ​**​1.9%​**​ respectively.

-----

## 🧠 Framework Overview

![](C:\Users\Administrator\Desktop\framework.jpg)

----

## 📊 Results

![](C:\Users\Administrator\AppData\Roaming\marktext\images\2025-09-08-19-33-04-image.png)

![](C:\Users\Administrator\AppData\Roaming\marktext\images\2025-09-08-19-32-58-image.png)

----

## 💾 Data Download

To run the BiFS framework, you need to download the following datasets:

- ​**​OK-VQA Dataset​**​: Available at [https://okvqa.allenai.org/](https://okvqa.allenai.org/)
- ​**​A-OKVQA Dataset​**​: Available at https://github.com/allenai/aokvqa

Please follow the respective websites' instructions for downloading and organizing these datasets.

-----

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

# 
