import lib
from common import parse_args, format_flops, format_mem
from collections import OrderedDict


runs = [
    "c4_small_sut_lowreg2_fixed_44m",
    "c4_sut_big_lessloss2_ffflopmatch_attmatch_actloss",
    "pes2o_sut_big_lessloss2_ffflopmatch_attmatch_actloss",
    "pes2o_small_sut_lowreg2_fixed_44m"
]


runs = [x for r in runs for x in lib.get_runs([r])]

def format_param_count(run):
    n = run.summary["n_params"] / 1e6
    return f"{int(round(n))}M"

def print_hyperparams(runs):
    for i, run in enumerate(runs):
        if i % 3 == 0 and i > 0:
            print("\\midrule")
        nparams = format_param_count(run)

        dmodel, nh, hps, dff, att_ne, att_k, nlayer, clip, ff_nheads, ut_gs, variant, ne, nwarmup, miloss, actloss, dexpert, dattexp = parse_args(run, "state_size", "transformer.n_heads", "transformer.head_projection_size", "dff", "moe.att.n_experts", "moe.att.k", "transformer.encoder_n_layers",  "grad_clip", "pkm.n_heads", "transformer.universal.group_size", "transformer.variant", "moe.n_experts", "lr_warmup", "moa.miloss", "transformer.act_loss", "moe.expert_size", "moe.att.expert_size")

        if variant != "actsut_universal":
            continue

        # SUT-specific
        nh = dattexp // hps

        print(f"{nparams} & {nlayer} & {dmodel} & {dexpert} & {nh} & {att_ne} & {dattexp} & {hps} & {ne} & {ff_nheads} & {miloss} & {actloss} & {nwarmup} & {clip} \\\\")

# 13
print("\\begin{tabular}{rrrrrrrrrrrrrr}")
print("\\toprule")
print("\\#params & $n_\\text{layers}$ & $d_\\text{model}$ & $\\dexpert$ & $H$ & $N_A$ & $d_\\text{att\\_expert}$ & $d_\\text{head}$ & $N_E$ & $K$ & $\\mathcal{L}_{\\text{MIM}}$ &  $\\mathcal{L}_{\\text{ACT}}$ & $N_\\text{warmup}$ & $\\kappa$ \\\\")
print("\\midrule")
print_hyperparams(runs)
print("\\bottomrule")
print("\\end{tabular}")