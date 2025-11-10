# LibMoE: A LIBRARY FOR COMPREHENSIVE BENCHMARKING MIXTURE OF EXPERTS IN LARGE LANGUAGE MODELS
**Authors:** Nam V. Nguyen*, Thong T. Doan*, Luong Tran, Van Nguyen, Quang Pham

<p align="center">
  <a href="https://arxiv.org/abs/2411.00918">
    <img src="https://img.shields.io/badge/arXiv-2411.00918-red?style=flat&label=Paper">
  </a>
  <a href="https://fsoft-aic.github.io/fsoft-LibMoE.github.io/">
    <img src="https://custom-icon-badges.demolab.com/badge/Webpage-1a4f76?style=flat&logo=web">
  </a>
</p>

<p align="center">
  <a href="#-quick-start">🚀 Quick Start</a> •
  <a href="#-overview">✨ Overview</a> •
  <a href="#-repository-map">🧱 Repository Map</a> •
  <a href="#-vision-language-stack">🖼️ Vision-Language</a> •
  <a href="#-language-modeling-stack">🧠 Language Modeling</a> •
  <a href="#-getting-started">🚦 Getting Started</a> •
  <a href="#-documentation-hub">📚 Docs</a> •
  <a href="#-release-notes">🗓️ Release Notes</a> •
  <a href="#-citation">📌 Citation</a>
</p>

---

## 📌 About
Mixture of experts (MoE) architectures have become a cornerstone for scaling up and are a key component in most large language models such as GPT-OSS, DeepSeek-V3, Llama-4, and Gemini-2.5. However, systematic research on MoE remains severely constrained by the prohibitive computational costs of training and evaluation, restricting large-scale studies accessible to most researchers. We introduce LibMoE, a unified framework for reproducible, efficient, and extensible MoE research that supports both pretraining and sparse-upcycling regimes. Beyond unified implementations, the framework provides transparent analytical tools for probing routing and expert dynamics. Leveraging this foundation, we conduct a comprehensive analysis along three dimensions: (i) routing dynamics, covering expert selection patterns, routing stability and optimality, and how routing entropy reveals task specialization and expert diversity; (ii) the effect of lightweight initialization on load balancing, demonstrating how subtle changes in router initialization shape early expert utilization; and (iii) training regime differences, revealing how sparse upcycling and full pretraining exhibit distinct routing patterns and stability profiles. By lowering the barrier to entry and standardizing evaluation, along with our comprehensive analysis, LibMoE broadens access to MoE research and establishes a reliable benchmark to guide future innovations.

## 📢 Release Notes

| Date       | Release Notes                                                                                              |
|------------|------------------------------------------------------------------------------------------------------------|
| 2024-12-30 | - Release LibMoE v1.1:  <br>   - Reduced training time by 70%, from approximately ~30h to ~9h.  <br>   - Provides more detailed information on MoE algorithms, including balancing loss, z-loss, training time per step, FLOPs, language loss, total loss, aux loss, and more customizable metrics. <br>   - Updated balance_loss_coef and router_z_loss_coef for better performance [More details](https://github.com/Fsoft-AIC/LibMoE/blob/main/scripts/train/phi35mini/siglip/sft.sh). |
| 2024-11-04 | - New feature: Metric analysis for MoE algorithms, as detailed in the [LibMoE](#) paper ✅ |
| 2024-11-01 | - Released LibMoE v1.0 preprint report: [Read Here](https://arxiv.org/pdf/2411.00918) ✅  <br>   - LibMoE webpage: [Visit Here](https://fsoft-aic.github.io/fsoft-LibMoE.github.io/) ✅  <br>   - Publicly available checkpoints ✅ |




## 🧱 Repository Map

```
LibMoEv2/
├── docs/
│   ├── pretrain_llm/
│   └── sparse_upcyling/
├── language_modeling/
│   ├── framework/
│   │   ├── data_structures/
│   │   ├── dataset/
│   │   ├── helpers/
│   │   ├── interfaces/
│   │   ├── layers/
│   │   ├── loader/
│   │   ├── optimizer/
│   │   ├── task/
│   │   └── utils/
│   ├── interfaces/
│   ├── layers/
│   │   └── transformer/
│   ├── models/
│   ├── paper/
│   │   ├── deepseek/
│   │   └── moe_universal/
│   ├── scripts/
│   ├── sweeps/
│   │   ├── 154M/
│   │   └── 660M/
│   └── tasks/
└── vision_language_model/
    ├── evaluate/
    │   ├── analysis/
    │   ├── docs/
    │   ├── lmms_eval/
    │   ├── miscs/
    │   ├── modules/
    │   ├── results/
    │   └── tools/
    ├── moe_model/
    │   ├── model/
    │   │   ├── language_model/
    │   │   ├── moe/
    │   │   ├── multimodal_encoder/
    │   │   └── multimodal_projector/
    │   ├── serve/
    │   │   └── examples/
    │   └── train/
    └── scripts/
        ├── eval/
        └── train/
```
---

## 🚀 Quick Start

### 1. Clone & Prepare Python (3.9 or 3.10)

```bash
git clone https://github.com/Fsoft-AIC/LibMoE.git
cd LibMoE
```

- `venv`

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

- `conda`

  ```bash
  conda create -n libmoe python=3.9 -y
  conda activate libmoe
  ```

### 2. Install the Stack Once

```bash
pip install --upgrade pip
pip install -e .
pip install -e .[vlm,lm,eval]          # or: pip install -r requirements.txt
```

Need a lighter environment? Start with `pip install -e .` and then layer on:

- Vision-language stack: `pip install -e .[vlm,eval]`
- Language-model pretraining: `pip install -e .[lm]`
- Evaluation utilities only: `pip install -e .[eval]`

### 3. (Optional) GPU Optimisations

Install FlashAttention that matches your CUDA/Torch stack:

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

Consult the [FlashAttention releases](https://github.com/Dao-AILab/flash-attention/releases) for alternate CUDA / PyTorch combinations.

---

## 🖼️ Vision-Language Stack

- **Pipeline** – automated pre-training, pre-finetuning, and visual instruction tuning (`vision_language_model/scripts/train/run_train_all.sh`).
- **Checkpoints** – SigLIP/Phi and CLIP/Phi releases across pre-train, pre-finetune, and SFT stages on Hugging Face.
- **Environment playbooks** – Python 3.9/3.10 setup, editable installs, FlashAttention guidance, and dataset preparation walkthroughs.
- **Evaluation suite** – AI2D, ChartQA, TextVQA, GQA, HallusionBenchmark, MathVista, MMBench, MME, MMMU, MMStar, POPE, and SQA IMG via `vision_language_model/scripts/eval/run_eval.sh`.
- **Analyst toolkit** – router entropy, expert overlap, and batch plotting documented in [`Analyst Tools README`](vision_language_model/evaluate/analysis/analyst_README.md).

➡️ Dive deeper: [`Vision-Language Stack Guide`](vision_language_model/vlm_README.md)

---

## 🧠 Language Modeling Stack

- **Configurable Transformer** – pluggable MoE layers in `language_modeling/layers/` and `language_modeling/layers/transformer/`.
- **MoE variants** – vanilla, X-MoE, DeepSeek-v2/v3, ReMoE, MoE++, TC-MoE selectable via `MOE_TYPE`.
- **Triton kernels** – sparse batched matmul (`language_modeling/layers/cvmm.py`) optimised for modern CUDA GPUs.
- **Streaming datasets** – SlimPajama ingestion with on-the-fly SentencePiece tokenisation and caching under `language_modeling/framework/dataset/text/`.
- **Task orchestration** – reusable dataset-model bindings in `language_modeling/tasks/` with YAML sweeps (`language_modeling/sweeps/154M`, `language_modeling/sweeps/660M`) and helper scripts (`language_modeling/scripts/train.sh`, `language_modeling/scripts/eval.sh`).

➡️ Dive deeper: [`Language Modeling Stack Guide`](language_modeling/LM_README.md)

---

## 📚 Documentation Hub

- **Vision-Language Stack**
  - [`Vision-Language Stack Guide`](vision_language_model/vlm_README.md) – complete multimodal workflow.
  - [`Evaluation Docs`](vision_language_model/evaluate/docs) – benchmark-specific evaluation notes and dataset references.
  - [`Analyst Tools README`](vision_language_model/evaluate/analysis/analyst_README.md) – router metrics, expert selection, plotting utilities.
- **Language Modeling Stack**
  - [`Language Modeling Stack Guide`](language_modeling/LM_README.md) – complete language pretraining workflow.
  - [`Model Guide`](docs/pretrain_llm/model_guide.md) – extend MoE layers, tune hyperparameters, integrate Triton kernels.
  - [`Dataset Guide`](docs/pretrain_llm/dataset_guide.md) – SlimPajama streaming, tokenisation, caching, custom datasets.
  - [`Checkpoint Catalogue`](docs/pretrain_llm/checkpoint_list.md) – language-model checkpoints and configurations.

---

## 📌 Citation

If this repository supports your research, please cite:

```
@misc{nguyen2025libmoelibrarycomprehensivebenchmarking,
      title={LIBMoE: A Library for comprehensive benchmarking Mixture of Experts in Large Language Models}, 
      author={Nam V. Nguyen and Thong T. Doan and Luong Tran and Van Nguyen and Quang Pham},
      year={2025},
      eprint={2411.00918},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.00918}, 
}
```
