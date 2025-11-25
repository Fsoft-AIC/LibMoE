def get_hps(run):
    state_size, n_heads = parse_args(run, "state_size", "transformer.n_heads")
    return int(state_size) // int(n_heads)

def get_dff(run):
    state_size, ffm = parse_args(run, "state_size", "transformer.ff_multiplier")
    return int(int(state_size) * float(ffm))


special_args = {
    "transformer.head_projection_size": get_hps,
    "dff": get_dff,
}

def parse_args(run, *args):
    res = []
    for a in args:
        found = None
        if a in run.config:
            found = run.config[a]
            # res.append(run.config[a])
        elif a in special_args:
            found = special_args[a](run)
            # res.append(special_args[a](run))
        else:
            assert False, f"Arg {a} not found"

        if isinstance(found, str) and found.lower()=="none" and a in special_args:
            found = special_args[a](run)

        res.append(found)
    return res

def get_attention_flops(run):
    trafo_var, att_var, state_size, n_heads, unroll, hps, cblocks, att_k, q_e, k_e, v_e, o_e, n_experts, att_en, n_att_groups = parse_args(run, "transformer.variant", "moe.att.variant", "state_size", "transformer.n_heads", "lm.unroll", "transformer.head_projection_size", "lm.trafo.context_blocks", "moe.att.k", "moe.att.q_expert", "moe.att.k_expert", "moe.att.v_expert", "moe.att.o_expert", "moe.att.n_experts", "moe.att.enable", "transformer.n_attention_groups")

    if trafo_var.endswith("_universal"):
        trafo_var = trafo_var[:-10]

    if trafo_var in "preln_moe":
        if att_en:
            trafo_var = "preln_moeatt"
        else:
            trafo_var = "preln_relative"

    state_size = int(state_size)
    n_heads = int(n_heads)
    unroll = int(unroll)
    hps = int(hps) if hps!="none" else (state_size // n_heads)
    cblocks = int(cblocks)
    att_k = int(att_k)
    n_experts = int(n_experts)

    if trafo_var == "preln_relative":
        return n_heads * hps * unroll ** 2 * (1+cblocks) * 2 + 4 * state_size * n_heads * hps * unroll + 2 * n_heads * state_size * unroll * hps * (1+cblocks)
    elif trafo_var == "preln_moeatt":
        if att_var in {"full", "full_rope"}:
            n_exp = int(q_e) + int(k_e) + int(v_e) + int(o_e)
            n_total_proj = n_exp * att_k + 4 - n_exp

            n_sels = min(1, int(k_e) + int(v_e)) + min(1, int(q_e) + int(o_e))

            res = n_heads * hps * unroll ** 2 * (1+cblocks) * 2 + n_total_proj * state_size * n_heads * hps * unroll + n_exp*n_heads*att_k*unroll*hps + n_sels * n_experts * state_size * unroll * (1+cblocks)
            if att_var == "full":
                res += 2 * n_heads * state_size * unroll * hps
            return res
        elif att_var == "moa":
            return n_heads * hps * unroll ** 2 * (1+cblocks) * 2 + (2*n_heads + 2) * state_size * hps * unroll  + n_heads*att_k*unroll*hps + 1 * n_experts * state_size * unroll + 2 * state_size * unroll * hps * (1+cblocks)
        else:
            assert False
    elif trafo_var == "preln_rope":
        return n_heads * hps * unroll ** 2 * (1+cblocks) * 2 + 4 * state_size * n_heads * hps * unroll
    else:
        assert False
    # print(state_size, n_heads, unroll, hps)


def get_ff_flops(run):
    trafo_var, state_size, unroll, ffmul, nexp, expsize, nheads = parse_args(run, "transformer.variant", "state_size", "lm.unroll", "transformer.ff_multiplier", "moe.n_experts", "moe.expert_size", "pkm.n_heads")
    state_size = int(state_size)
    unroll = int(unroll)
    ffmul = float(ffmul)
    nexp = int(nexp)
    expsize = int(expsize)
    nheads = int(nheads)

    if trafo_var.endswith("_universal"):
        trafo_var = trafo_var[:-10]

    if trafo_var in {"preln_relative", "preln_rope"}:
        return 2 * state_size * int(state_size * ffmul) * unroll
    elif trafo_var == "preln_moe":
        return 2 * unroll * state_size * expsize * nheads + unroll * state_size * nexp
    else:
        raise ValueError(f"Unknown transformer variant {trafo_var}")


def get_flops(run):
    n_layers, = parse_args(run, "transformer.encoder_n_layers")
    n_layers = int(n_layers)
    return (get_attention_flops(run) + get_ff_flops(run))*n_layers


def get_train_flops(run):
    stop_after, batch_size = parse_args(run, "stop_after", "batch_size")
    stop_after = int(stop_after)
    batch_size = int(batch_size)
    return get_flops(run) * stop_after * batch_size


def get_attention_mem(run):
    trafo_var, att_var, state_size, n_heads, unroll, hps, cblocks, att_k, q_e, k_e, v_e, o_e, n_experts, att_en, n_att_groups = parse_args(run, "transformer.variant", "moe.att.variant", "state_size", "transformer.n_heads", "lm.unroll", "transformer.head_projection_size", "lm.trafo.context_blocks", "moe.att.k", "moe.att.q_expert", "moe.att.k_expert", "moe.att.v_expert", "moe.att.o_expert", "moe.att.n_experts", "moe.att.enable", "transformer.n_attention_groups")

    if trafo_var == "preln_moe":
        if att_en:
            trafo_var = "preln_moeatt"
        else:
            trafo_var = "preln_relative"

    state_size = int(state_size)
    n_heads = int(n_heads)
    unroll = int(unroll)
    hps = int(hps) if hps!="none" else (state_size // n_heads)
    cblocks = int(cblocks)
    att_k = int(att_k)
    n_experts = int(n_experts)

    if trafo_var in "preln_relative":
        return n_heads * unroll ** 2 * (1+cblocks) * 2 + 4 * n_heads * hps * unroll + 2 * n_heads * unroll * hps * (1+cblocks)
    elif trafo_var == "preln_frel_group":
        return n_heads * unroll ** 2 * (1+cblocks) * 2 + \
               2 * (n_heads + n_att_groups) * hps * unroll + \
               2 * n_att_groups * unroll * hps * (1+cblocks)
    elif trafo_var == "preln_moeatt":
        if att_var in {"full", "full_rope"}:
            # K doesn't matter for the memory usage (with a smart kernel)
            n_sels = min(1, int(k_e) + int(v_e)) + min(1, int(q_e) + int(o_e))

            res = n_heads * unroll ** 2 * (1+cblocks) * 2 + 4 * n_heads * hps * unroll + n_sels * n_experts * unroll * (1+cblocks)
            if att_var == "full":
                res += 2 * n_heads * unroll * hps
            return res
        elif att_var == "moa":
            return n_heads * unroll ** 2 * (1+cblocks) * 2 + (2*n_heads + 2) * hps * unroll + 2 * unroll * hps + 1 * n_experts * unroll * (1+cblocks)
        else:
            assert False
    elif trafo_var == "preln_rope":
        return n_heads * unroll ** 2 * (1+cblocks) * 2 + 4 * n_heads * hps * unroll
    else:
        assert False
    # print(state_size, n_heads, unroll, hps)


def format_flops(run):
    flp = get_attention_flops(run)/1e6

    if flp > 1000:
        return f"{flp/1000:.1f}G"
    else:
        return f"{flp:.1f}M"


def format_mem(run):
    flp = get_attention_mem(run)/1e6

    if flp > 1000:
        return f"{flp/1000:.1f}G"
    else:
        return f"{flp:.1f}M"

