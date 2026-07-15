"""Build runs/results_log.html -- a single self-contained gallery of every
stage's result figures with plain-English reminders of what each stage is.

Rerun after any stage completes: python experiments/_common/build_results_log.py
Add a stage = add an entry to STAGES below. Images are embedded as data URIs
so the page is fully self-contained (required by the Artifact CSP).
"""

from __future__ import annotations

import base64
import datetime
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "results_log.html"

# Each stage: status is "passed" | "caveat" | "pending".
STAGES = [
    {
        "id": "Stage 0",
        "title": "Baseline model & calibration",
        "status": "passed",
        "what": (
            "Train the evidential CNN on clean (unfiltered) rotated sevens with true "
            "angle labels. This produces the pretrained model every later stage starts "
            "from. Two gates: it must be accurate, and its uncertainty must be honest "
            "(higher uncertainty on the samples it gets more wrong)."
        ),
        "reading": (
            "Left: training loss. Middle: held-out circular MSE during training. "
            "Right: each dot is one held-out image -- predicted uncertainty vs actual "
            "error; the upward trend is the calibration signal."
        ),
        "numbers": [
            ("Held-out MSE", "0.0123 ± 0.0020 (~20.0°)"),
            ("Uncertainty–error corr.", "≈ 0.72"),
            ("Reference", "0.0106 (~18.5°), corr one-draw 0.5649"),
            ("Verdict", "consistent within training variance (seeds 42/43/44 spanned 0.012–0.019; old checkpoint scores 0.0101 under identical eval)"),
        ],
        "image": ROOT / "runs/stage0/baseline/figures/stage0_results.png",
        "manifest": "runs/stage0/baseline/manifest.yaml",
    },
    {
        "id": "Stage 1",
        "title": "Degradation under distributional shift",
        "status": "passed",
        "what": (
            "Take the Stage 0 model, train it no further, and evaluate it as an "
            "increasingly strong red filter is applied to the images. Shows the model "
            "genuinely breaks under shift -- so adaptation is necessary, not optional."
        ),
        "reading": (
            "Left: MSE vs filter strength (error bars = spread over 3 eval passes). "
            "Right: the model's own mean uncertainty vs filter strength -- it rises "
            "with the shift despite never being told about filters."
        ),
        "numbers": [
            ("Degradation", "0.0126 → 0.1205 (9.6×), monotonic at all 10 steps"),
            ("Uncertainty rise", "0.0088 → 0.0586 (6.7×), also monotonic"),
        ],
        "image": ROOT / "runs/stage1/red_degradation/figures/stage1_results.png",
        "manifest": "runs/stage1/red_degradation/manifest.yaml",
    },
]

STAGES += [
    {
        "id": "Stage 2a",
        "title": "Peer supervision (sequential)",
        "status": "caveat",
        "what": (
            "At each red filter strength, Node A fine-tunes on true angle labels, then "
            "Node B trains ONLY on A's predictions -- no ground truth ever reaches B. "
            "Tests whether a peer's output is a sufficient training signal."
        ),
        "reading": (
            "Where the filter is strong (right side), A and B sit together far below "
            "the un-adapted baseline: peer supervision works. At the far left both are "
            "WORSE than the baseline -- after adapting to red they've forgotten the "
            "clean domain (all weights are trainable; expected, and the old repo's own "
            "simultaneous figure showed the same). B forgot less than A."
        ),
        "numbers": [
            ("B vs A tracking gap", "mean |B−A| = 0.0085"),
            ("At full red", "A 0.0088, B 0.0121, un-adapted 0.1224"),
            ("Curve smoothness (std)", "B 0.0189 vs A 0.0299 -- B smoother, as reference predicts"),
            ("Caveat", "at strengths 0.0–0.2 both A and B trail the un-adapted baseline (clean-domain forgetting)"),
        ],
        "image": ROOT / "runs/stage2/sequential/figures/stage2_sequential.png",
        "manifest": "runs/stage2/sequential/manifest.yaml",
    },
    {
        "id": "Stage 2b",
        "title": "Peer supervision (simultaneous)",
        "status": "caveat",
        "what": (
            "Same pairing, but the red strength ramps 0 to 1 across 20 epochs while A and B "
            "train at the same time -- does peer supervision still work when the teacher "
            "is itself still adapting?"
        ),
        "reading": (
            "Left: training losses (note B's loss is measured against A's outputs, not "
            "truth, so the curves aren't directly comparable). Right: final performance -- "
            "B lands near A under shift and far below the un-adapted baseline. The "
            "reference expected B's loss to lag A's early; ours didn't (old run's lag was "
            "within run-to-run variance)."
        ),
        "numbers": [
            ("At full red", "A 0.0081, B 0.0109, un-adapted 0.1200"),
            ("Early loss lag", "not observed (B−A = −0.07 first 3 epochs)"),
        ],
        "image": ROOT / "runs/stage2/simultaneous/figures/stage2_simultaneous.png",
        "manifest": "runs/stage2/simultaneous/manifest.yaml",
    },
]

STAGES += [
    {
        "id": "Stage 3",
        "title": "Naive aggregation baseline",
        "status": "caveat",
        "what": (
            "Node A adapts to red, Node C to blue (both with true labels, strength "
            "ramping 0 to 1). Node B -- living in red -- trains only on the plain average "
            "of A's and C's predictions. C is the mismatched anchor: everything B sees is "
            "out-of-distribution for it. Establishes the cost of averaging anchors of "
            "unequal reliability, which uncertainty-aware fusion exists to avoid."
        ),
        "reading": (
            "Bottom-left is the key panel: B (orange) sits between the good anchor A and "
            "the bad anchor C at every meaningful strength -- naive averaging drags B away "
            "from the anchor that actually knows the domain. Bottom-right: the logged "
            "uncertainty summary did NOT show C elevated above A (deviation from the old "
            "run); the asymmetry does exist in full predictive width at high strengths "
            "(C 1.2-1.4x wider than A), which is the quantity fusion consumes."
        ),
        "numbers": [
            ("At full red", "A 0.0100 < B 0.0208 < C 0.0465, un-adapted 0.1126"),
            ("B between A and C", "yes, at all strengths ≥ 0.4"),
            ("C vs A, Student-t width", "1.20× @ 0.8, 1.36× @ 1.0 (weaker than old run; inverts below 0.4)"),
            ("Extra finding", "B's uncertainty collapses to ~0.0006 -- smooth averaged pseudo-labels breed overconfidence"),
        ],
        "image": ROOT / "runs/stage3/naive_average/figures/stage3_results.png",
        "manifest": "runs/stage3/naive_average/manifest.yaml",
    },
]

STAGES += [
    {
        "id": "Stage 4a",
        "title": "Scalar weighting strategies",
        "status": "caveat",
        "what": (
            "Six ways of averaging A's and C's predictions using only a collapsed "
            "(prediction, certainty) pair per anchor: naive, certainty-weighted, squared, "
            "gated, softmax, winner-only. All share the limitation fusion removes: they "
            "throw away the distribution and keep two numbers."
        ),
        "reading": (
            "Right panel: all six cluster tightly above Node A. Winner-only is best -- "
            "but only because B's filter matches A's, so 'pick the more certain anchor' "
            "degenerates to 'always pick A'. Gated is NOT worst here (old result was "
            "plausibly a silent bug in its fallback, fixed in V2). Scalar gains over naive "
            "are muted, consistent with Stage 3's weaker uncertainty asymmetry."
        ),
        "numbers": [
            ("High-strength ranking", "winner 0.0146 < squared/softmax 0.0211 < gated 0.0223 < certainty 0.0233 < naive 0.0238"),
            ("Node A / un-adapted", "0.0119 / 0.0750"),
        ],
        "image": ROOT / "runs/stage4/4a_scalar_weighting/figures/stage4_4a_scalar_weighting.png",
        "manifest": "runs/stage4/4a_scalar_weighting/manifest.yaml",
    },
    {
        "id": "Stage 4b",
        "title": "Bayesian fusion vs naive averaging",
        "status": "passed",
        "what": (
            "The corrected product-of-experts fusion (wrapped, G=360) against naive "
            "averaging, same anchors, same protocol. The stage the whole correction "
            "was for: does consuming the full belief distributions beat any scalar "
            "summary of them?"
        ),
        "reading": (
            "Fusion (orange) tracks Node A closely across the sweep and pulls far below "
            "naive at high strengths. At strength 0.8, fusion matches A itself despite "
            "receiving zero ground truth. The gap to Stage 4a's scalar cluster is the "
            "evidence that density SHAPE, not a collapsed reliability score, carries "
            "the signal."
        ),
        "numbers": [
            ("High-strength mean", "fusion 0.0132 vs naive 0.0224 (41% lower)"),
            ("At strength 0.8", "fusion 0.0110 ≈ Node A 0.0107; un-adapted 0.0699"),
            ("vs best legitimate scalar", "0.0132 vs ~0.0211 (37% lower)"),
        ],
        "image": ROOT / "runs/stage4/4b_fusion_vs_naive/figures/stage4_4b_fusion_vs_naive.png",
        "manifest": "runs/stage4/4b_fusion_vs_naive/manifest.yaml",
    },
    {
        "id": "Stage 4c",
        "title": "Grid vs optimisation fusion",
        "status": "passed",
        "what": (
            "Two ways of finding the mode of the fused density: grid search (G=360 "
            "candidates) vs gradient descent. If both work comparably, the gain comes "
            "from principled mode estimation itself, not the particular estimator -- "
            "and grid search wins on deployment grounds (predictable cost)."
        ),
        "reading": (
            "Grid and optimisation sit nearly on top of each other, optimisation "
            "marginally ahead, both far below naive -- exactly the spec's expected "
            "pattern. (The N=2 bit-exactness hard gate lives in the test suite, "
            "tests/test_2_pairwise_control.py, and passed separately.)"
        ),
        "numbers": [
            ("High-strength mean", "grid 0.0161, optim 0.0151, naive 0.0262"),
        ],
        "image": ROOT / "runs/stage4/4c_grid_vs_optim/figures/stage4_4c_grid_vs_optim.png",
        "manifest": "runs/stage4/4c_grid_vs_optim/manifest.yaml",
    },
]

ROADMAP = [
    ("Stage 5", "Variable environments: B's conditions drift between red and blue (linear / oscillating / random) -- fusion vs naive averaging."),
    ("Stage 6", "Spatial mesh: 36 overlapping nodes, moving wearable anchor, belief propagation. 6D is the headline three-way comparison."),
]


def img_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def build() -> str:
    updated = datetime.date.today().isoformat()

    sections = []
    for s in STAGES:
        chip_class = {"passed": "chip-pass", "caveat": "chip-caveat", "pending": "chip-pend"}[s["status"]]
        chip_text = {"passed": "gate passed", "caveat": "passed w/ caveat", "pending": "pending"}[s["status"]]
        numbers = "".join(
            f'<div class="num"><dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd></div>'
            for k, v in s["numbers"]
        )
        sections.append(f"""
<section>
  <p class="eyebrow">{html.escape(s["id"])} <span class="chip {chip_class}">{chip_text}</span></p>
  <h2>{html.escape(s["title"])}</h2>
  <p class="what">{html.escape(s["what"])}</p>
  <figure>
    <div class="imgwrap"><img src="{img_data_uri(s["image"])}" alt="{html.escape(s["id"])} result charts"></div>
    <figcaption>{html.escape(s["reading"])}</figcaption>
  </figure>
  <dl class="numbers">{numbers}</dl>
  <p class="manifest">manifest: <code>{html.escape(s["manifest"])}</code></p>
</section>""")

    roadmap_items = "".join(
        f'<div class="road"><dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd></div>'
        for k, v in ROADMAP
    )

    return f"""<title>BeliefMesh Results Log</title>
<style>
:root {{
  --bg: #f7f8f6; --surface: #ffffff; --ink: #1c2420; --muted: #5c6660;
  --accent: #2e7d4f; --line: #dfe4df; --code-bg: #eef1ee;
  --pass-bg: #e3efe7; --pass-ink: #22603d;
  --pend-bg: #ececec; --pend-ink: #666;
  --caveat-bg: #f3ecdd; --caveat-ink: #7a5c1e;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #141815; --surface: #1c221e; --ink: #e4e9e4; --muted: #93a097;
    --accent: #57b57e; --line: #2a322c; --code-bg: #232b26;
    --pass-bg: #1e3528; --pass-ink: #7fce9f;
    --pend-bg: #262b28; --pend-ink: #8a938c;
    --caveat-bg: #33301f; --caveat-ink: #cfb06a;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #141815; --surface: #1c221e; --ink: #e4e9e4; --muted: #93a097;
  --accent: #57b57e; --line: #2a322c; --code-bg: #232b26;
  --pass-bg: #1e3528; --pass-ink: #7fce9f;
  --pend-bg: #262b28; --pend-ink: #8a938c;
  --caveat-bg: #33301f; --caveat-ink: #cfb06a;
}}
:root[data-theme="light"] {{
  --bg: #f7f8f6; --surface: #ffffff; --ink: #1c2420; --muted: #5c6660;
  --accent: #2e7d4f; --line: #dfe4df; --code-bg: #eef1ee;
  --pass-bg: #e3efe7; --pass-ink: #22603d;
  --pend-bg: #ececec; --pend-ink: #666;
  --caveat-bg: #f3ecdd; --caveat-ink: #7a5c1e;
}}
html {{ background: var(--bg); }}
body {{
  font-family: Charter, Georgia, "Times New Roman", serif;
  color: var(--ink); line-height: 1.55;
  max-width: 900px; margin: 0 auto; padding: 3rem 1.25rem 5rem;
}}
header {{ border-bottom: 2px solid var(--accent); padding-bottom: 1.25rem; margin-bottom: 2.5rem; }}
h1 {{ font-size: 1.9rem; margin: 0 0 .4rem; text-wrap: balance; }}
.sub {{ color: var(--muted); margin: 0; max-width: 65ch; }}
.eyebrow {{
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-size: .78rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--accent); margin: 0 0 .3rem; display: flex; align-items: center; gap: .6rem;
}}
.chip {{
  font-size: .68rem; letter-spacing: .08em; padding: .15rem .55rem;
  border-radius: 2px; text-transform: uppercase;
}}
.chip-pass {{ background: var(--pass-bg); color: var(--pass-ink); }}
.chip-pend {{ background: var(--pend-bg); color: var(--pend-ink); }}
.chip-caveat {{ background: var(--caveat-bg); color: var(--caveat-ink); }}
section {{ margin-bottom: 3.25rem; }}
h2 {{ font-size: 1.35rem; margin: 0 0 .6rem; text-wrap: balance; }}
.what {{ max-width: 68ch; margin: 0 0 1.1rem; }}
figure {{ margin: 0 0 1rem; background: var(--surface); border: 1px solid var(--line); border-radius: 4px; padding: .75rem; }}
.imgwrap {{ overflow-x: auto; }}
.imgwrap img {{ display: block; max-width: 100%; height: auto; }}
figcaption {{ font-size: .85rem; color: var(--muted); padding: .6rem .25rem 0; max-width: 75ch; }}
dl.numbers {{ display: flex; flex-wrap: wrap; gap: .75rem 2rem; margin: 0 0 .5rem; }}
.num dt {{
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted);
}}
.num dd {{ margin: .1rem 0 0; font-variant-numeric: tabular-nums; max-width: 46ch; }}
.manifest {{ font-size: .8rem; color: var(--muted); }}
code {{
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-size: .85em; background: var(--code-bg); padding: .1em .35em; border-radius: 3px;
}}
.roadmap {{ border-top: 1px solid var(--line); padding-top: 1.75rem; }}
.roadmap dl {{ display: grid; gap: .9rem; margin: 1rem 0 0; }}
.road {{ display: grid; grid-template-columns: 6.5rem 1fr; gap: 1rem; }}
.road dt {{
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-size: .78rem; letter-spacing: .1em; text-transform: uppercase;
  color: var(--pend-ink); padding-top: .15rem;
}}
.road dd {{ margin: 0; color: var(--muted); max-width: 68ch; }}
</style>

<header>
  <p class="eyebrow">BeliefMesh · Experiments V2</p>
  <h1>Results Log</h1>
  <p class="sub">One chart section per completed stage, with a reminder of what each stage
  actually tests. Regenerated as stages land. Updated {updated}.</p>
</header>

{"".join(sections)}

<div class="roadmap">
  <p class="eyebrow">Still to run</p>
  <dl>{roadmap_items}</dl>
</div>
"""


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
