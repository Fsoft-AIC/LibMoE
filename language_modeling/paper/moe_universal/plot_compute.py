import lib
import datetime

runs = lib.get_runs([
    "c4_baseline_big_rope_nodrop", "c4_moeut_big_matched_rope_preln_linear_norot_nosm1",
    "c4_baseline_bigger_rope_nodrop", "c4_moeut_bigger_matched_rope_nonorm_g3",
    "c4_baseline_small_rope_long_nodrop", "c4_moeut_small_matched_rope_noln_long",
    "c4_moeut_gigantic_switchead", "c4_baseline_gigantic_rope_nodrop",
    "c4_moeut_mid_matched_rope_noln", "c4_baseline_mid_rope_nodrop",

    "c4_moeut_big_matched_rope_noln_g18", "c4_moeut_mid_matched_rope_noln_g18",
    "c4_moeut_bigger_matched_rope_nonorm_g24", "c4_moeut_gigantic_switchead_g36",
    "c4_moeut_small_matched_rope_noln_long_g16",

    "c4_baseline_1b_rope_nodrop", "c4_moeut_1b_switchead",
    "pes2o_moeut_small_matched_rope_noln_long", "pes2o_baseline_small_rope_long_nodrop",
    "pes2o_baseline_big_rope_nodrop", "pes2o_moeut_big_matched_rope_noln",

    "slimpajama_baseline_small_rope_long_nodrop", "slimpajama_baseline_big_rope_nodrop",
    "slimpajama_moeut_small_matched_rope_noln_long", "slimpajama_moeut_big_matched_rope_noln",
    "slimpajama_moeut_1b_switchead", "slimpajama_baseline_1b_rope_nodrop",

    "c4_small_sut_lowreg2_fixed_44m",

    "c4_sut_big_lessloss2_ffflopmatch_attmatch_actloss"

    # All PES2O big
    "pes2o_sut_big_lessloss2_ffflopmatch_attmatch_actloss",

    # All PES2O small
    "pes2o_small_sut_lowreg2_fixed_44m",

    "thestack_baseline_big_rope_nodrop",
    "thestack_moeut_big_matched_rope_noln",

    "thestack_moeut_gigantic_switchead_multilang",
    "thestack_baseline_gigantic_rope_nodrop_multilang",


    "c4_moeut_big_matched_rope_preln",
    "c4_moeut_big_matched_rope_postln",

    "c4_baseline_small_rope_long_nodrop",
    "c4_moeut_small_matched_rope_postln",

    "c4_moeut_big_matched_rope_noln_g1",
    "c4_moeut_big_matched_rope_noln_g3",
    "c4_moeut_big_matched_rope_noln_g6",
    "c4_moeut_big_matched_rope_noln_g9",
    "c4_moeut_big_matched_rope_noln_g18",
    "c4_moeut_big_matched_rope_noln_aabb",
])

runs = list(sorted(runs, key=lambda r: (r.config["transformer.variant"], r.summary["n_params"], r.config["task"], r.config["transformer.universal.group_size"])))

datasets = {
    "c4_transformer": "C4",
    "slimpajama_transformer": "SlimPajama",
    "pes2o_transformer": "peS2o",
    "thestack_transformer": "TheStack"
}

def get_ngpus(r):
    logs = lib.get_logs(r)
    for l in logs:
        h = "World size: "
        hi = l.find(h)
        if hi > 0:
            return int(l[hi+len(h):].split(",")[0])
        elif "No distributed environment detected" in l:
            return 1

    return "??"
    raise ValueError("Number of GPUs not found")

variants = {
    "preln_rope": "Transformer",
    "preln_moe_universal": "MoEUT",
    "moe_universal": "MoEUT PostLN",
    "actsut_universal": "SUT",
}

def get_variant(r):
    if r.config["transformer.variant"] == "preln_moe_universal" and r.config["transformer.universal.group_size"] == r.config["transformer.encoder_n_layers"] and r.config["moe.att.n_experts"]==1:
        return "$\\sigma$-MoE"

    res = variants[r.config["transformer.variant"]]
    if r.config["transformer.variant"] == "preln_moe_universal" and not r.config.get("moe.nonorm", False):
        res += " PreLN"

    if r.config["transformer.variant"] == "preln_moe_universal" and r.config.get("transformer.universal.group_type", "abab") == "aabb":
        res += " AABB"

    return res

def format_duration(duration):
    hours = duration // 3600
    minutes = (duration % 3600) // 60

    # Format it as a string in hh:mm:ss
    return f"{hours}:{minutes:02}"

for r in runs:
    ngpus = get_ngpus(r)
    ncpus = r.metadata["cpu_count"]
    ram = r.metadata["memory"]["total"]
    runtime = r.summary["_wandb"].runtime
    gpu_type = r.metadata["gpu"]

    gpu_type = gpu_type.replace("NVIDIA ", "").replace("-SXM2", "").replace("-SXM4", "").replace("-PCIE", "").replace("GeForce ", "").replace("Tesla ", "")

    variant = get_variant(r)
    dataset = datasets[r.config["task"]]
    nparams = int(round(r.summary["n_params"] / 1e6))
    group_size = r.config["transformer.universal.group_size"]

    if "MoEUT" not in variant:
        group_size = "-"

    print(f"{variant} & {nparams}M & {dataset} & {group_size} & {gpu_type} & {ngpus} & {ncpus} & {int(ram)}G & {format_duration(runtime)} \\\\")



