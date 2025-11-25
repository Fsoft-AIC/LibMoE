import lib
from common import parse_args, format_flops, format_mem
from collections import OrderedDict


runs = [
    "c4_baseline_small_rope_long_nodrop",
    "c4_moeut_small_matched_rope_noln_long",
    "c4_moeut_small_matched_rope_noln_long_g16",

    "c4_baseline_mid_rope_nodrop",
    "c4_moeut_mid_matched_rope_noln",
    "c4_moeut_mid_matched_rope_noln_g18",

    "c4_baseline_big_rope_nodrop",
    "c4_moeut_big_matched_rope_preln_linear_norot_nosm1",
    "c4_moeut_big_matched_rope_noln_g18",

    "c4_baseline_bigger_rope_nodrop",
    "c4_moeut_bigger_matched_rope_nonorm_g3",
    "c4_moeut_bigger_matched_rope_nonorm_g24",

    "c4_baseline_gigantic_rope_nodrop",
    "c4_moeut_gigantic_switchead",
    "c4_moeut_gigantic_switchead_g36",

    "c4_baseline_1b_rope_nodrop",
    "c4_moeut_1b_switchead"
]

variant_names = {
    "preln_rope": "Baseline",
    "preln_moe_universal": "MoEUT",
}

runs = [
    lib.get_runs([r])[0] for r in runs
]

def format_param_count(run):
    n = run.summary["n_params"] / 1e6
    return f"{int(round(n))}M"

def print_hyperparams(runs):
    for i, run in enumerate(runs):
        if i % 3 == 0 and i > 0:
            print("\\midrule")
        nparams = format_param_count(run)
        # Even better. Sometimes it does not upload the config! If missing, try parsing the command line.

        dmodel, nh, hps, dff, att_ne, att_k, nlayer, clip, ff_nheads, ut_gs, variant, ne, nwarmup= parse_args(run, "state_size", "transformer.n_heads", "transformer.head_projection_size", "dff", "moe.att.n_experts", "moe.att.k", "transformer.encoder_n_layers",  "grad_clip", "pkm.n_heads", "transformer.universal.group_size", "transformer.variant", "moe.n_experts", "lr_warmup")
        varname = variant_names[variant]

        if variant == "preln_moe_universal":
            dff = "-"
            if ut_gs == nlayer and att_ne==1:
                varname = "$\\sigma$-MoE"
        else:
            ne = "-"
            ut_gs = "-"
            ff_nheads = "-"
            att_ne = "-"

        print(f"{varname} & {nparams} & {nlayer} & {ut_gs} & {dmodel} & {dff}   & {nh} & {att_ne} & {hps} & {ne} & {ff_nheads} & {nwarmup} & {clip} \\\\")

# 13
print("\\begin{tabular}{lrrrrrrrrrrrr}")
print("\\toprule")
print("Model & \\#params & $n_\\text{layers}$ & $G$ & $d_\\text{model}$ & $d_\\text{ff}$ & $H$ & $N_A$ & $d_\\text{head}$ & $N_E$ & $K$ & $N_\\text{warmup}$ & $\\kappa$ \\\\")
print("\\midrule")
print_hyperparams(runs)
print("\\bottomrule")
print("\\end{tabular}")