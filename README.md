
<!-- 📌 Banner -->
<p align="center">
  <img width="1536" height="500" alt="LibMoE Banner" src="https://github.com/user-attachments/assets/f1c9e5b2-82fe-4cdb-816a-d7dabba2fa15" />
</p>

<!-- 📌 Row: Webpage + arXiv -->
<p align="center">
  <a href="https://fsoft-aic.github.io/fsoft-LibMoE.github.io/">
    <img src="https://custom-icon-badges.demolab.com/badge/Webpage-1a4f76?style=flat&logo=web" alt="Webpage"/>
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2411.00918">
    <img src="https://img.shields.io/badge/arXiv-2411.00918-red?style=flat&label=Paper" alt="arXiv"/>
  </a>
</p>

<!-- 📌 Title -->
<h1 align="center">
  LibMoE: A LIBRARY FOR COMPREHENSIVE BENCHMARKING MIXTURE OF EXPERTS IN LARGE LANGUAGE MODELS
</h1>

<!-- 📌 Authors -->
<p align="center">
  <b>Authors:</b> Nam V. Nguyen*, Thong T. Doan*, Luong Tran, Van Nguyen, Quang Pham
</p>

## 📌 About
Mixture of experts (MoE) architectures have become a cornerstone for scaling up and are a key component in most large language models such as GPT-OSS, DeepSeek-V3, Llama-4, and Gemini-2.5. However, systematic research on MoE remains severely constrained by the prohibitive computational costs of training and evaluation, restricting large-scale studies accessible to most researchers. We introduce LibMoE, a unified framework for reproducible, efficient, and extensible MoE research that supports both pretraining and sparse-upcycling regimes. Beyond unified implementations, the framework provides transparent analytical tools for probing routing and expert dynamics. Leveraging this foundation, we conduct a comprehensive analysis along three dimensions: (i) routing dynamics, covering expert selection patterns, routing stability and optimality, and how routing entropy reveals task specialization and expert diversity; (ii) the effect of lightweight initialization on load balancing, demonstrating how subtle changes in router initialization shape early expert utilization; and (iii) training regime differences, revealing how sparse upcycling and full pretraining exhibit distinct routing patterns and stability profiles. By lowering the barrier to entry and standardizing evaluation, along with our comprehensive analysis, LibMoE broadens access to MoE research and establishes a reliable benchmark to guide future innovations.

## 📢 Release Notes

| Date       | Release Notes                                                                                              |
|------------|------------------------------------------------------------------------------------------------------------|
| 2024-12-30 | - Release LibMoE v1.1:  <br>   - Reduced training time by 70%, from approximately ~30h to ~9h.  <br>   - Provides more detailed information on MoE algorithms, including balancing loss, z-loss, training time per step, FLOPs, language loss, total loss, aux loss, and more customizable metrics. <br>   - Updated balance_loss_coef and router_z_loss_coef for better performance [More details](https://github.com/Fsoft-AIC/LibMoE/blob/main/scripts/train/phi35mini/siglip/sft.sh). |
| 2024-11-04 | - New feature: Metric analysis for MoE algorithms, as detailed in the [LibMoE](#) paper ✅ |
| 2024-11-01 | - Released LibMoE v1.0 preprint report: [Read Here](https://arxiv.org/pdf/2411.00918) ✅  <br>   - LibMoE webpage: [Visit Here](https://fsoft-aic.github.io/fsoft-LibMoE.github.io/) ✅  <br>   - Publicly available checkpoints ✅ |


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

After installing all required libraries, follow the component-specific guides below:

---

🖼️ **Vision-Language Stack — Sparse Upcycling**

LibMoE provides a streamlined **sparse-upcycling pipeline**, converting existing VLM backbones (SigLIP/CLIP × Phi) into MoE-enhanced architectures without training from scratch. The pipeline supports pre-training, pre-fine-tuning, and visual instruction tuning.

➡️ [`Vision-Language Guide`](vision_language_model/vlm_README.md)

---

🧠 **Language Modeling Stack — MoE Pretraining from Scratch**

The language modeling stack focuses on **end-to-end MoE pretraining from scratch**, featuring a modular Transformer design, flexible routing strategies, and a suite of MoE variants for comprehensive sparse LLM research.

➡️ [`Language Modeling Guide`](language_modeling/LM_README.md)

---
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
