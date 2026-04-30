"""Auto-generated module extracted from visual_data.ipynb."""
from __future__ import annotations

def run() -> None:
    category_map_tmp = {
        # ─────────── Math / Reasoning ───────────
        "CLEVR-Math(MathV360K)": "Math/Reasoning",
        "FigureQA(MathV360K)":   "Math/Reasoning",
        "GEOS(MathV360K)":       "Math/Reasoning",
        "GeoQA+(MathV360K)":     "Math/Reasoning",
        "Geometry3K(MathV360K)": "Math/Reasoning",
        "IconQA(MathV360K)":     "Math/Reasoning",
        "MapQA(MathV360K)":      "Math/Reasoning",
        "MathV360K_TQA":         "Math/Reasoning",
        "MathV360K_VQA-AS":      "Math/Reasoning",
        "MathV360K_VQA-RAD":     "Math/Reasoning",
        "PMC-VQA(MathV360K)":    "Math/Reasoning",
        "Super-CLEVR(MathV360K)": "Math/Reasoning",
        "TabMWP(MathV360K)":     "Math/Reasoning",
        "UniGeo(MathV360K)":     "Math/Reasoning",
        "VizWiz(MathV360K)":     "Math/Reasoning",
        "geo170k(align)":        "Math/Reasoning",
        "geo170k(qa)":           "Math/Reasoning",
        "geo3k":                 "Math/Reasoning",
        "geomverse(cauldron)":   "Math/Reasoning",
        "mathqa":                "Math/Reasoning",
        "mavis_math_metagen":    "Math/Reasoning",
        "mavis_math_rule_geo":   "Math/Reasoning",
        "raven(cauldron)":       "Math/Reasoning",
        "iconqa(cauldron,llava_format)": "Math/Reasoning",
        "mapqa(cauldron,llava_format)":  "Math/Reasoning",

        # ─────────── Doc / Chart / Screen ───────────
        "ai2d(cauldron,llava_format)": "Doc/Chart/Screen",
        "ai2d(gpt4v)":                "Doc/Chart/Screen",
        "ai2d(internvl)":             "Doc/Chart/Screen",
        "chart2text(cauldron)":       "Doc/Chart/Screen",
        "chartqa(cauldron,llava_format)": "Doc/Chart/Screen",
        "diagram_image_to_text(cauldron)": "Doc/Chart/Screen",
        "dvqa(cauldron,llava_format)":     "Doc/Chart/Screen",
        "figureqa(cauldron,llava_format)": "Doc/Chart/Screen",
        "hitab(cauldron,llava_format)":    "Doc/Chart/Screen",
        "infographic(gpt4v)":              "Doc/Chart/Screen",
        "infographic_vqa":                 "Doc/Chart/Screen",
        "infographic_vqa_llava_format":    "Doc/Chart/Screen",
        "lrv_chart":                       "Doc/Chart/Screen",
        "lrv_normal(filtered)":            "Doc/Chart/Screen",
        "robut_sqa(cauldron)":             "Doc/Chart/Screen",
        "robut_wikisql(cauldron)":         "Doc/Chart/Screen",
        "robut_wtq(cauldron,llava_format)": "Doc/Chart/Screen",
        "screen2words(cauldron)":          "Doc/Chart/Screen",
        "visualmrc(cauldron)":             "Doc/Chart/Screen",
        "ureader_cap":                     "Doc/Chart/Screen",
        "ureader_ie":                      "Doc/Chart/Screen",
        "tqa(cauldron,llava_format)": "Doc/Chart/Screen",

        # ─────────── General OCR ───────────
        "chrome_writting":          "General OCR",
        "iam(cauldron)":            "General OCR",
        "iiit5k":                   "General OCR",
        "k12_printing":             "General OCR",
        "rendered_text(cauldron)":  "General OCR",
        "textcaps":                 "General OCR",
        "textocr(gpt4v)":           "General OCR",
        "hme100k":                  "General OCR",
        "websight(cauldron)":       "General OCR",

        # ─────────── General ───────────
        "allava_instruct_laion4v":      "General",
        "allava_instruct_vflan4v":      "General",
        "aokvqa(cauldron,llava_format)": "General",
        "image_textualization(filtered)": "General",
        "intergps(cauldron,llava_format)": "General",
        "llava_wild_4v_12k_filtered":   "General",
        "llava_wild_4v_39k_filtered":   "General",
        "llavar_gpt4":                  "General",
        "hateful_memes(cauldron,llava_format)": "General",
        "scienceqa(cauldron,llava_format)":     "General",
        "sharegpt4o":                   "General",
        "sharegpt4v(coco)":             "General",
        "sharegpt4v(llava)":            "General",
        "st_vqa(cauldron,llava_format)": "General",
        "tallyqa(cauldron,llava_format)": "General",
        "vistext(cauldron)":           "General",
        "visual7w(cauldron,llava_format)": "General",
        "vqarad(cauldron,llava_format)":   "General",
        "vsr(cauldron,llava_format)":      "General",
            "CLEVR(cauldron,llava_format)":  "General",  # clevr …


        # ─────────── Language (chưa có trong list) ───────────
        # "Magpie …": "Language",
    }

    category_map = {}

    for k, v in category_map_tmp.items():
        category_map[k.lower()] = v

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import to_rgb, to_hex
    import numpy as np, json, math
    plt.rcParams['font.family'] = 'DejaVu Serif'
    fontsize = 19
    plt.ioff()  # faster off‑screen rendering

    # ============ 1) DATA RAW (same counts) ============ #
    # /cm/archive/anonymous/data/jsons/onevison600K/cag_stats_sampled.json

    with open("/cm/archive/anonymous/data/jsons/onevison600K/cag_stats_sampled.json", "r") as f:

        data = json.load(f)

    # ============ 2) PROFESSIONAL DISPLAY NAMES ============ #
    dataset_name = {
        "CLEVR-Math(MathV360K)": "CLEVR‑Math (MathV360K)",
        "FigureQA(MathV360K)": "FigureQA (MathV360K)",
        "GEOS(MathV360K)": "GEOS (MathV360K)",
        "GeoQA+(MathV360K)": "GeoQA+ (MathV360K)",
        "Geometry3K(MathV360K)": "Geometry3K (MathV360K)",
        "IconQA(MathV360K)": "IconQA (MathV360K)",
        "MapQA(MathV360K)": "MapQA (MathV360K)",
        "MathV360K_TQA": "TQA (MathV360K)",
        "MathV360K_VQA-AS": "VQA‑AS (MathV360K)",
        "MathV360K_VQA-RAD": "VQA‑RAD (MathV360K)",
        "PMC-VQA(MathV360K)": "PMC‑VQA (MathV360K)",
        "Super-CLEVR(MathV360K)": "Super‑CLEVR (MathV360K)",
        "TabMWP(MathV360K)": "TabMWP (MathV360K)",
        "UniGeo(MathV360K)": "UniGeo (MathV360K)",
        "VizWiz(MathV360K)": "VizWiz (MathV360K)",
        "ai2d(cauldron,llava_format)": "AI2D (Cauldron/LLaVA)",
        "ai2d(gpt4v)": "AI2D (GPT‑4V)",
        "ai2d(internvl)": "AI2D (InternVL)",
        "allava_instruct_laion4v": "ALLaVA‑Instr. (LAION‑4V)",
        "allava_instruct_vflan4v": "ALLaVA‑Instr. (V‑FLAN‑4V)",
        "aokvqa(cauldron,llava_format)": "AOK‑VQA (Cauldron/LLaVA)",
        "chart2text(cauldron)": "Chart2Text (Cauldron)",
        "chartqa(cauldron,llava_format)": "ChartQA (Cauldron/LLaVA)",
        "chrome_writting": "Chrome Writing",
        "clevr(cauldron,llava_format)": "CLEVR (Cauldron/LLaVA)",
        "diagram_image_to_text(cauldron)": "Diagram2Text (Cauldron)",
        "dvqa(cauldron,llava_format)": "DVQA (Cauldron/LLaVA)",
        "figureqa(cauldron,llava_format)": "FigureQA (Cauldron/LLaVA)",
        "geo170k(align)": "Geo170K‑Align",
        "geo170k(qa)": "Geo170K‑QA",
        "geo3k": "Geo3K",
        "geomverse(cauldron)": "GeoMVerse (Cauldron)",
        "hateful_memes(cauldron,llava_format)": "Hateful Memes (Cauldron/LLaVA)",
        "hitab(cauldron,llava_format)": "HiTab (Cauldron/LLaVA)",
        "hme100k": "HME‑100K",
        "iam(cauldron)": "IAM (Cauldron)",
        "iconqa(cauldron,llava_format)": "IconQA (Cauldron/LLaVA)",
        "iiit5k": "IIIT‑5K",
        "image_textualization(filtered)": "Image Textualization (Filt.)",
        "infographic(gpt4v)": "Infographic (GPT‑4V)",
        "infographic_vqa": "Infographic‑VQA",
        "infographic_vqa_llava_format": "Infographic‑VQA (LLaVA)",
        "intergps(cauldron,llava_format)": "InterGPS (Cauldron/LLaVA)",
        "k12_printing": "K‑12 Printing",
        "llava_wild_4v_12k_filtered": "LLaVA‑Wild 12K",
        "llava_wild_4v_39k_filtered": "LLaVA‑Wild 39K",
        "llavar_gpt4": "LLaVAR (GPT‑4)",
        "lrv_chart": "LRV‑Chart",
        "lrv_normal(filtered)": "LRV‑Normal (Filt.)",
        "mapqa(cauldron,llava_format)": "MapQA (Cauldron/LLaVA)",
        "mathqa": "MathQA",
        "mavis_math_metagen": "MAVIS – MetaGen",
        "mavis_math_rule_geo": "MAVIS – RuleGeo",
        "raven(cauldron)": "Raven (Cauldron)",
        "rendered_text(cauldron)": "Rendered Text (Cauldron)",
        "robut_sqa(cauldron)": "RoBUT‑SQA (Cauldron)",
        "robut_wikisql(cauldron)": "RoBUT‑WikiSQL (Cauldron)",
        "robut_wtq(cauldron,llava_format)": "RoBUT‑WTQ (Cauldron/LLaVA)",
        "scienceqa(cauldron,llava_format)": "ScienceQA (Cauldron/LLaVA)",
        "screen2words(cauldron)": "Screen2Words (Cauldron)",
        "sharegpt4o": "ShareGPT‑4o",
        "sharegpt4v(coco)": "ShareGPT‑4V (COCO)",
        "sharegpt4v(llava)": "ShareGPT‑4V (LLaVA)",
        "st_vqa(cauldron,llava_format)": "ST‑VQA (Cauldron/LLaVA)",
        "tallyqa(cauldron,llava_format)": "TallyQA (Cauldron/LLaVA)",
        "textcaps": "TextCaps",
        "textocr(gpt4v)": "TextOCR (GPT‑4V)",
        "tqa(cauldron,llava_format)": "TQA (Cauldron/LLaVA)",
        "ureader_cap": "UReader‑Cap",
        "ureader_ie": "UReader‑IE",
        "vistext(cauldron)": "VisText (Cauldron)",
        "visual7w(cauldron,llava_format)": "Visual7W (Cauldron/LLaVA)",
        "visualmrc(cauldron)": "VisualMRC (Cauldron)",
        "vqarad(cauldron,llava_format)": "VQARAD (Cauldron/LLaVA)",
        "vsr(cauldron,llava_format)": "VSR (Cauldron/LLaVA)",
        "websight(cauldron)": "WebSight (Cauldron)"
    }
    disp = lambda k: dataset_name.get(k, k)

    # ============ 3) CATEGORY, COLORS ============ #
    def categorize(n: str):
        return category_map[n.lower()]
    def fmt_count(n):
        """Trả về chuỗi gọn gàng cho n mẫu."""
        if n >= 1_000_000:          # ≥ 1 M
            return f"{n/1_000_000:.1f} M"
        elif n >= 1_000:            # 1 K – 999 K
            return f"{n/1_000:.1f} K"
        else:                       # < 1 K
            return str(n)
    BASE_COL = {"General":"#f28e1c","Doc/Chart/Screen":"#55a6ff",
                "Math/Reasoning":"#55b9a6","General OCR":"#8bc34a"}
    def shade(col, idx, n):
        t = 0.1 + 0.8*idx/(n-1 or 1)
        rgb = np.array(to_rgb(col)); return to_hex(rgb + (1-rgb)*t)

    # ============ 4) GROUP & SORT ============ #
    groups={}
    for ds, cnt in data.items():
        groups.setdefault(categorize(ds), []).append((ds,cnt))
    for g in groups:
        groups[g].sort(key=lambda x:x[1], reverse=True)
    order=["General","Doc/Chart/Screen","Math/Reasoning","General OCR"]
    total=sum(data.values())

    # Prepare sunburst arrays
    inner_l, inner_s, inner_c = [],[],[]
    outer_s, outer_c = [],[]
    for g in order:
        if g not in groups: continue
        lst=groups[g]
        inner_l.append(g); inner_s.append(sum(c for _,c in lst)); inner_c.append(BASE_COL[g])
        for i,(ds,cnt) in enumerate(lst):
            outer_s.append(cnt); outer_c.append(shade(BASE_COL[g], i, len(lst)))

    # ============ 5) PLOT ============ #
    fig = plt.figure(figsize=(20,9))
    gs  = GridSpec(1,2, width_ratios=[1.1,1.3], wspace=0.05)

    # Donut
    ax=fig.add_subplot(gs[0])
    ax.pie(outer_s, radius=1.0, colors=outer_c, startangle=90,
           wedgeprops=dict(width=0.28, edgecolor='white'))
    inner_w,_ = ax.pie(inner_s, radius=0.72, colors=inner_c, startangle=90,
                       wedgeprops=dict(width=0.30, edgecolor='white'))

    for w,lbl,sz in zip(inner_w, inner_l, inner_s):
        theta=0.5*(w.theta1+w.theta2); rad=np.deg2rad(theta)
        label_r=0.55
        rot=theta+90
        if rot>180: rot-=180
        ax.text(label_r*np.cos(rad), label_r*np.sin(rad),
                f"{lbl}\n{sz/total*100:.1f}%", ha='center', va='center',
                rotation=rot, rotation_mode='anchor', fontsize=15)
    ax.text(0,0, f"One-Vision\n{total/1e6:.1f} M", ha='center', va='center', fontsize=17)
    ax.axis('equal')

    # Legend
    # ───── 5-b) LEGEND (4 cột, auto row_h & step_x) ─────
    axL = fig.add_subplot(gs[1]); axL.axis("off")

    col_n   = 4
    x_left  = 0.02
    step_x  = (0.97 - x_left) / (col_n - 1)        # ≤ 0.316, vừa khung trục

    # ---- ❶  tính tổng dòng cần hiển thị ----
    total_line_units = 0.0
    for g in order:
        if g not in groups: continue
        rows = (len(groups[g]) + col_n - 1) // col_n
        total_line_units += 1.4        # header
        total_line_units += rows       # mỗi hàng dataset = 1 đơn vị
        total_line_units += 0.4        # separator

    row_h = min(0.05,   0.95 / total_line_units)   # co giãn để không vượt trục

    # ---- ❷  vẽ legend với row_h mới ----
    y = 1.0
    for g in order:
        if g not in groups: continue
        col = BASE_COL[g]
        pct = sum(c for _,c in groups[g]) / total * 100

        axL.text(0.0, y, f"{g} ({pct:.1f}%)",
                 color="white", weight="bold", fontsize=14,
                 bbox=dict(facecolor=col, pad=3, edgecolor="none"),
                 transform=axL.transAxes)
        y -= row_h * 1.4

        lst  = groups[g]
        rows = (len(lst) + col_n - 1) // col_n
        for r in range(rows):
            for c in range(col_n):
                idx = r * col_n + c
                if idx >= len(lst): break
                ds, cnt = lst[idx]

                x = x_left + c * step_x
                axL.add_patch(Rectangle(
                    (x, y - 0.013), 0.018, 0.018,
                    facecolor=shade(col, idx, len(lst)),
                    transform=axL.transAxes
                ))
                axL.text(
                    x + 0.022, y,
                    f"{disp(ds)} ({fmt_count(cnt)})",
                    fontsize=10, va="center",
                    transform=axL.transAxes
                )
            y -= row_h

        # axL.plot([0.0, 0.97], [y + row_h/2]*2,
        #          color="black", lw=1, transform=axL.transAxes)
        y -= row_h * 0.4

    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03)

    fig.savefig("benchmarks_vlm.pdf", format="pdf", bbox_inches="tight")


    plt.show()

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import to_rgb, to_hex
    import numpy as np

    # ────── 0) CÀI ĐẶT CHUNG ──────
    plt.rcParams["font.family"] = "DejaVu Serif"
    fs = 16                              # cỡ chữ chung

    # ────── 1) DATA LLaVA-665K ──────
    llava665k_info = {
        "LLaVA":    {"actual": 157_712, "cag": "General"},
        "SG40k":    {"actual": 40_688,  "cag": "General"},
        "VQA-v2":   {"actual": 82_783,  "cag": "General"},
        "GQA":      {"actual": 72_140,  "cag": "Math/Reasoning"},
        "OKVQA":    {"actual": 8_998,   "cag": "General"},
        "OCRVQA":   {"actual": 80_000,  "cag": "General OCR"},
        "A-OKVQA":  {"actual": 66_160,  "cag": "General"},
        "TextCaps": {"actual": 21_953,  "cag": "General OCR"},
        "RefCOCO":  {"actual": 48_447,  "cag": "General"},
        "VG":       {"actual": 86_417,  "cag": "Math/Reasoning"},
    }

    def fmt_count(n):
        if n >= 1_000_000: return f"{n/1_000_000:.1f} M"
        if n >= 1_000:     return f"{n/1_000:.1f} K"
        return str(n)

    # ────── 2) COLOR & GROUP ──────
    BASE = {"General":"#f28e1c",
            "Doc/Chart/Screen":"#55a6ff",
            "Math/Reasoning":"#55b9a6",
            "General OCR":"#8bc34a"}

    def shade(col, i, n):                 # chuyển sắc dịu 25 → 75 %
        t = 0.25 + 0.50 * i / (n-1 or 1)
        return to_hex(np.array(to_rgb(col)) + (1 - np.array(to_rgb(col))) * t)

    groups = {}
    for ds, info in llava665k_info.items():
        groups.setdefault(info["cag"], []).append((ds, info["actual"]))
    for g in groups:
        groups[g].sort(key=lambda x: x[1], reverse=True)

    order = ["General", "Doc/Chart/Screen", "Math/Reasoning", "General OCR"]
    total = sum(v["actual"] for v in llava665k_info.values())

    # ────── 3) SUNBURST ARRAYS ──────
    inn_l, inn_s, inn_c = [], [], []
    out_s, out_c        = [], []
    for g in order:
        if g not in groups: continue
        inn_l.append(g)
        inn_s.append(sum(c for _, c in groups[g]))
        inn_c.append(BASE[g])
        for i, (ds, cnt) in enumerate(groups[g]):
            out_s.append(cnt)
            out_c.append(shade(BASE[g], i, len(groups[g])))

    # ────── 4) FIGURE (RẤT SÁT VIỀN) ──────
    fig = plt.figure(figsize=(18, 8))                       # nhỏ gọn hơn
    gs  = GridSpec(1, 2, width_ratios=[0.8, 1.15], wspace=0.05)

    # 4-a) Donut
    ax = fig.add_subplot(gs[0])
    ax.pie(out_s, radius=1.0, colors=out_c, startangle=90,
           wedgeprops=dict(width=0.28, edgecolor="white"))
    inn_w, _ = ax.pie(inn_s, radius=0.72, colors=inn_c, startangle=90,
                      wedgeprops=dict(width=0.30, edgecolor="white"))

    for w, lbl, sz in zip(inn_w, inn_l, inn_s):
        th  = 0.5*(w.theta1 + w.theta2)
        rot = th+90 if th+90 <= 180 else th-90
        ax.text(0.56*np.cos(np.deg2rad(th)),
                0.56*np.sin(np.deg2rad(th)),
                f"{lbl}\n{sz/total*100:.1f} %",
                ha="center", va="center",
                fontsize=fs, rotation=rot, rotation_mode="anchor")
    ax.text(0, 0, f"LLaVA-665K\n{total/1e6:.1f} M",
            ha="center", va="center", fontsize=fs+2)
    ax.axis("equal")

    # 4-b) Legend (4 cột, hẹp ngang)
    axL = fig.add_subplot(gs[1]); axL.axis("off")
    col_n, x_left, h_scale = 4, 0.02, 0.70            # h_scale<1 → sát ngang hơn
    step_x = h_scale * (0.97 - x_left) / (col_n - 1)
    row_h  = 0.055                                    # hẹp dọc hơn

    # tính chiều cao legend & căn giữa
    lines = 0
    for g in order:
        if g not in groups: continue
        lines += 1.4 + (len(groups[g]) + col_n - 1)//col_n + 0.4
    legend_h = lines * row_h
    y = 1 - max(0, 1 - legend_h) / 2

    for g in order:
        if g not in groups: continue
        col = BASE[g]
        pct = sum(c for _, c in groups[g]) / total * 100
        axL.text(0, y, f"{g} ({pct:.1f} %)",
                 color="white", weight="bold", fontsize=fs+1,
                 bbox=dict(facecolor=col, pad=3, edgecolor="none"),
                 transform=axL.transAxes)
        y -= row_h * 1.4

        lst  = groups[g]
        rows = (len(lst) + col_n - 1) // col_n
        for r in range(rows):
            for c in range(col_n):
                idx = r*col_n + c
                if idx >= len(lst): break
                ds, cnt = lst[idx]
                x = x_left + c * step_x
                axL.add_patch(Rectangle((x, y-0.014), 0.02, 0.02,
                                        facecolor=shade(col, idx, len(lst)),
                                        transform=axL.transAxes))
                axL.text(x + 0.024, y, f"{ds} ({fmt_count(cnt)})",
                         fontsize=fs-2, va="center", transform=axL.transAxes)
            y -= row_h

        # axL.plot([0, 0.97], [y + row_h/2]*2, color="black", lw=1,
        #          transform=axL.transAxes)
        y -= row_h * 0.4

    # Lề sát mép
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01)
    fig.tight_layout(pad=0.05)

    # Lưu PDF
    fig.savefig("benchmarks_llava665k.pdf", format="pdf")
    plt.show()

    category_map

    data['clevr(cauldron,llava_format)']


if __name__ == "__main__":
    run()
