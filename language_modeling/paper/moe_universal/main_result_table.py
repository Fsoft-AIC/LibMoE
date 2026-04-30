import lib
import math
import json

runs = lib.get_runs([
    "c4_baseline_big_rope_nodrop", "c4_moeut_big_matched_rope_preln_linear_norot_nosm1",
    "c4_baseline_bigger_rope_nodrop", "c4_moeut_bigger_matched_rope_nonorm_g3",
    "c4_baseline_small_rope_long_nodrop", "c4_moeut_small_matched_rope_noln_long",
    "c4_moeut_gigantic_switchead", "c4_baseline_gigantic_rope_nodrop",
    "c4_moeut_mid_matched_rope_noln", "c4_baseline_mid_rope_nodrop",
    "c4_baseline_1b_rope_nodrop", "c4_moeut_1b_switchead",

    "pes2o_moeut_small_matched_rope_noln_long", "pes2o_baseline_small_rope_long_nodrop",
    "pes2o_baseline_big_rope_nodrop", "pes2o_moeut_big_matched_rope_noln",

    "slimpajama_baseline_small_rope_long_nodrop", "slimpajama_baseline_big_rope_nodrop",
    "slimpajama_moeut_small_matched_rope_noln_long", "slimpajama_moeut_big_matched_rope_noln",
    "slimpajama_moeut_1b_switchead", "slimpajama_baseline_1b_rope_nodrop"
])

print(runs)


def select_closest(val, options, tolerance=10):
    res = min(options, key=lambda x: abs(x-val))
    if abs(res-val) > tolerance:
        raise ValueError(f"Closest value {res} is too far from {val}")
    return res


columns = [
    {
        "type": "config",
        "field": "task",
        "transform": lambda x: {
            "c4_transformer": "C4",
            "pes2o_transformer": "PES2O",
            "slimpajama_transformer": "SlimPajama"
        }[x],
        "break": True,
        "merge_identical": True,
        "align": "center"
    },

    {
        "type": "summary_config",
        "field": "n_params",
        "transform": lambda x: select_closest(int(round(x/1e6)), {44, 126, 244, 319, 728, 1040}),
        "format": lambda x: f"{x}M",
        "break": True,
        "merge_identical": True,
        "align": "center"
    },

    {
        "type": "config",
        "field": "transformer.variant",
        "transform": lambda x: {
            "preln_rope": "Baseline",
            "preln_moe_universal": "MoEUT",
        }[x],
    },

    {
        "type": "summary",
        "field": "validation/val/perplexity",
        "transform": lambda x: f"{x:.2f}"
    },

    {
        "type": "summary",
        "field": "validation/lambada/accuracy/total",
        "transform": lambda x: f"{x*100:.1f}\\%",
        "average": True
    },

    {
        "type": "summary",
        "field": "validation/blimp/accuracy/group_average",
        "transform": lambda x: f"{x*100:.1f}\\%",
        "average": True
    },

    {
        "type": "summary",
        "field": "validation/cbt/accuracy/group_average",
        "transform": lambda x: f"{x*100:.1f}\\%",
        "average": True
    },

    {
        "type": "summary",
        "field": "validation/hellaswag/accuracy/group_average",
        "transform": lambda x: f"{x*100:.1f}\\%",
        "average": True
    },

    {
        "type": "summary",
        "field": "validation/piqa/accuracy/group_average",
        "transform": lambda x: f"{x*100:.1f}\\%",
        "average": True
    },

    {
        "type": "summary",
        "field": "validation/ai2arc/accuracy/ARC-Easy",
        "transform": lambda x: f"{x*100:.1f}\\%",
        "average": True
    },

    {
        "type": "summary",
        "field": "average",
        "transform": lambda x: f"{x*100:.1f}\\%",
    },


    # {
    #     "type": "summary",
    #     "field": "validation/ai2arc/accuracy/ARC-Challenge",
    #     "transform": lambda x: f"{x*100:.1f}\\%"
    # },

    # {
    #     "type": "summary",
    #     "field": "validation/winogrande/accuracy/group_average",
    #     "transform": lambda x: f"{x*100:.1f}\\%"
    # },
]

summary_cache = {}

def get_summary(run):
    global summary_cache

    cached = summary_cache.get(run.id)
    if cached is not None:
        return cached

    news = dict(run.summary)
    with open(f"checkpoints/{run.id}/result.json", "r") as f:
        testres = json.load(f)

    with open(f"checkpoints/{run.id}/result2.json", "r") as f:
        testres.update(json.load(f))

    with open(f"checkpoints/{run.id}/result3.json", "r") as f:
        testres.update(json.load(f))

    testres = {f"validation/{k}": v for k, v in testres.items()}
    news.update(testres)

    summary_cache[run.id] = news
    return news



def render_table(runs, columns):
    options = {}

    getters = {
        "config": lambda r: r.config,
        "summary_config": get_summary,
        "summary": get_summary
    }

    for i, c in enumerate(columns):
        if c["type"] in {"config", "summary_config"}:
            get_src = getters[c["type"]]
            transform = c.get("transform", lambda x: x)
            options[i] = list(sorted(set([transform(get_src(r)[c["field"]]) for r in runs])))


    opt_order = list(sorted(options.keys()))

    counts = {i: len(v) for i, v in options.items()}
    for ii in range(len(opt_order)-2, -1, -1):
        ithis = opt_order[ii]
        inext = opt_order[ii+1]
        counts[ithis] = counts[inext] * counts[ithis]

    strides = {opt_order[ii]: counts[ii+1] for ii in range(len(opt_order)-1)}
    strides[opt_order[-1]] = 1

    rows = []
    for row in range(counts[opt_order[0]]):
        state = {}
        for i, opt in enumerate(opt_order):
            state[opt] = (row // strides[opt]) % len(options[opt])

        runs_ok = list(runs)
        for i in opt_order:
            c = columns[i]
            if c["type"] in {"config", "summary_config"}:
                get_src = getters[c["type"]]
                transform = c.get("transform", lambda x: x)
                runs_ok = [r for r in runs_ok if transform(get_src(r)[c["field"]]) == options[i][state[i]]]

        if not runs_ok:
            continue

        if len(runs_ok) > 1:
            raise ValueError(f"Multiple runs for row {row}")

        col = []
        avg_sum = 0
        avg_cnt = 0
        for c in columns:
            transform = c.get("transform", lambda x: x)
            format = c.get("format", lambda x: x)

            if c["field"] == "average":
                col.append(format(transform(avg_sum / avg_cnt)) if avg_cnt > 0 else "-")
                continue

            get_src = getters[c["type"]]
            data = get_src(runs_ok[0])[c["field"]]
            if c.get("average", False):
                avg_sum += data
                avg_cnt += 1
            col.append(format(transform(data)))


        rows.append(col)

    first_of_its_kind = [[True] * len(columns)] + [[False] * len(columns) for _ in range(len(rows)-1)]
    for i in range(1, len(rows)):
        for j in range(len(columns)):
            first_of_its_kind[i][j] = rows[i][j] != rows[i-1][j]

    identical_counts = [[1] * len(columns) for _ in range(len(rows))]
    for i in range(len(rows)-2, -1, -1):
        for j in range(len(columns)):
            next_c = 0 if first_of_its_kind[i+1][j] else identical_counts[i+1][j]
            identical_counts[i][j] = next_c + 1

    for i, r in enumerate(rows):
        break_from = None

        srow = ""
        for j, c in enumerate(columns):
            merge = c.get("merge_identical", False)
            if merge:
                if first_of_its_kind[i][j]:
                    if c.get("align", "center") == "center" and identical_counts[i][j] > 1:
                        srow += "\\multirow{" + str(identical_counts[i][j]) + "}{*}{" + rows[i][j] + "}"
                    else:
                        srow += rows[i][j]
            else:
                srow += rows[i][j]

            if break_from is None and c.get("break", False) and i > 0 and first_of_its_kind[i][j]:
                break_from = j

            srow += " & "

        if break_from == 0:
            print("\\midrule")
        elif break_from is not None:
            print("\\cmidrule(lr){" +str(break_from + 1) + "-" + str(len(columns)) + "}")


        print(srow[:-3] + " \\\\")

        # \multirow{3}{*}{C4}

print("Rendering table...")
render_table(runs, columns)