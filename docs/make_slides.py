"""Build docs/presentation.pptx from the saved figures (no training needed).

    python docs/make_slides.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
OUT = ROOT / "docs" / "presentation.pptx"

DARK = RGBColor(0x1F, 0x2A, 0x44)
ACCENT = RGBColor(0xD5, 0x5E, 0x00)
GREY = RGBColor(0x55, 0x5B, 0x66)
W, H = Inches(13.333), Inches(7.5)


def add_text(slide, text, left, top, width, height, size=18, bold=False, color=DARK):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text if isinstance(text, list) else [text]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, color
        p.space_after = Pt(6)
    return box


def slide(prs, title, subtitle=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
    add_text(s, title, Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.7), size=28, bold=True)
    if subtitle:
        add_text(s, subtitle, Inches(0.5), Inches(0.9), Inches(12.3), Inches(0.5), size=15, color=GREY)
    return s


def picture(s, name, left, top, max_w, max_h):
    """Insert an image scaled to fit inside the (max_w, max_h) box, centred horizontally."""
    from PIL import Image
    path = FIG / name
    with Image.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    pw, ph = int(w * scale), int(h * scale)
    s.shapes.add_picture(str(path), Emu(left + (max_w - pw) // 2), Emu(top), Emu(pw), Emu(ph))


def bullets_and_figure(prs, title, subtitle, bullets, fig, split=0.32):
    s = slide(prs, title, subtitle)
    add_text(s, [f"• {b}" for b in bullets], Inches(0.5), Inches(1.6), int(W * split), Inches(5.5), size=16)
    picture(s, fig, int(W * split) + Inches(0.6), Inches(1.5), int(W * (1 - split)) - Inches(1.0), Inches(5.8))
    return s


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1. title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_text(s, "Tabular Reinforcement Learning", Inches(0.8), Inches(2.3), Inches(11.5), Inches(1), size=44, bold=True)
    add_text(s, "SARSA & Q-learning on the class gridworld, plus comparison of 6 algorithms, Cliff Walking, "
                "Optuna tuning and training analysis", Inches(0.8), Inches(3.4), Inches(11.5), Inches(1), size=20, color=GREY)
    add_text(s, "Igor Ruiz & Andoni Garrido · Reinforcement Learning · Universidad de Deusto · October 2026",
             Inches(0.8), Inches(5.6), Inches(11.5), Inches(0.5), size=16, color=ACCENT)

    # 2. setup
    bullets_and_figure(prs, "Setup", "Same environment as in class (3x4 gridworld), plus a second one (Cliff Walking)", [
        "Gridworld 3x4: +1 goal, −1 pit, wall; deterministic and slippery (80% intended, 10%/10% sideways)",
        "Cliff Walking 4x12: −1 per step, −100 on the cliff (back to start)",
        "Agents are model-free: they only call reset()/step()",
        "Value iteration (uses P) only gives the ground truth Q* for the metrics",
        "20 seeds per configuration, ε-greedy with decay, γ = 0.99",
    ], "exp1_values_policies.png", split=0.34)

    # 3. SARSA vs Q-learning: update rules and learning curves
    bullets_and_figure(prs, "Minimum requirement: SARSA vs Q-learning", "Both learn the optimal policy in both gridworlds", [
        "SARSA (on-policy, Bellman eq.):  Q ← Q + α[r + γ Q(s',a') − Q]",
        "Q-learning (off-policy, BOE):  Q ← Q + α[r + γ max Q(s',·) − Q]",
        "Deterministic: π* in all 20 seeds (Q-learning stable after ~10 episodes, SARSA ~200)",
        "Slippery, default schedule (ε_min = 0.05, constant α): Q-learning 19/20 seeds optimal, SARSA 15/20",
        "SARSA learns Q of the ε-greedy policy and rarely visits the states next to the pit",
        "Theory fix: GLIE (ε → 0) + Robbins–Monro α(s,a) = α/(1+k·N) → both 20/20 seeds optimal",
    ], "exp1_learning_curves.png", split=0.3)

    # 4. all algorithms
    s = slide(prs, "Comparison of all tabular algorithms",
              "Monte Carlo · SARSA · n-step SARSA · Expected SARSA · Q-learning · Double Q-learning")
    picture(s, "exp2_summary_bars.png", Inches(0.4), Inches(1.5), W - Inches(0.8), Inches(3.6))
    add_text(s, ["• Q-learning has the most accurate Q* and the lowest regret overall",
                 "• Expected SARSA is the best on-policy method (lower variance than SARSA)",
                 "• Monte Carlo fails in Cliff Walking: episodes rarely finish, so returns are uninformative (needs terminal episodes)"],
             Inches(0.6), Inches(5.3), Inches(12), Inches(2), size=16)

    # 5. cliff
    bullets_and_figure(prs, "On-policy vs off-policy: Cliff Walking", "Classic Sutton & Barto experiment, ε = 0.1, α = 0.5", [
        "Q-learning learns the optimal path on the cliff edge, but falls while exploring",
        "SARSA learns a safer path because its target includes its own random actions",
        "Online return: SARSA better · Greedy policy: Q-learning optimal",
    ], "exp3_cliff_walking.png", split=0.3)

    # 6. training analysis
    s = slide(prs, "Analysis of the training process", "Exploration schedule and Monte Carlo vs TD bias/variance")
    picture(s, "exp5b_exploration.png", Inches(0.4), Inches(1.4), W - Inches(0.8), Inches(2.8))
    picture(s, "exp5c_bias_variance.png", Inches(2.2), Inches(4.3), W - Inches(4.4), Inches(3.1))

    # 7. optuna: full multi-objective study if it has been run (experiments.tune), else the preliminary one
    if (FIG / "exp4_default_vs_tuned.png").exists():
        bullets_and_figure(prs, "Hyper-parameter tuning with Optuna", "Multi-objective TPE: speed vs exactness, 6 algorithms x 2 environments", [
            "Search: α, Robbins–Monro decay of α, ε schedule (start, min, half-life), n for n-step",
            "Two objectives: mean regret during training (speed) and final regret (exactness) → Pareto front",
            "Tuned on 5 seeds, compared on 20 held-out seeds",
            "Bootstrap 95% CI + permutation test (* p<0.05)",
            "See results/tables/exp4_default_vs_tuned.csv for the numbers",
        ], "exp4_default_vs_tuned.png", split=0.28)
    else:
        bullets_and_figure(prs, "Hyper-parameter tuning with Optuna", "TPE, 40 trials, objective = area under the regret curve", [
            "Tuned α and the ε schedule (start, min, decay) on 5 seeds",
            "Held-out (20 new seeds): mean regret SARSA 0.036 → 0.014, Q-learning 0.015 → 0.007",
            "Q-learning tuned: 19/20 seeds optimal (default 7/20)",
            "Over-fitting is visible: SARSA objective 0.001 on tuning seeds but 0.014 held-out",
            "Lower regret ≠ more exact: tuned SARSA learns faster, but only 5/20 seeds reach exactly π*",
        ], "exp4_preliminary_optuna.png", split=0.28)

    # 8. conclusions
    s = slide(prs, "Conclusions")
    add_text(s, [
        "• SARSA and Q-learning both solve the deterministic and the stochastic gridworld",
        "• Q-learning (BOE) → Q*; SARSA (BE) → Q of the ε-greedy policy: safer behaviour while exploring",
        "• Stochastic transitions need decaying α and ε → 0 (Robbins–Monro + GLIE) to reach π* in every seed",
        "• TD bootstraps (low variance, biased); MC uses real returns (unbiased, high variance, needs episodes to end)",
        "• Expected SARSA lowers variance; Double Q removes maximisation bias but learns more slowly",
        "• Everything is reproducible: results cached; `python -m experiments.run_all` redraws without training",
    ], Inches(0.7), Inches(1.4), Inches(12), Inches(5.5), size=20)

    # backup: failure analysis
    bullets_and_figure(prs, "Backup: why MC and Double Q fail in Cliff Walking", "Estimates collapse to the −1/(1−γ) = −100 'never arrive' plateau", [
        "MC with 1/N: the −1500 returns of the first episodes stay in the mean forever (non-stationary target)",
        "Constant-α MC: 14/20 stuck seeds → 0/20; exploring starts helps further",
        "Double Q, α = 0.5: 3/20 seeds collapse to −100; α = 0.1 → 0/20",
    ], "exp6_failure_analysis.png", split=0.3)

    # 9-10. backup slides for the questions (not part of the 5-minute talk)
    s = slide(prs, "Backup: dynamic programming (model-based)", "Used only as ground truth; agents never see P")
    add_text(s, [
        "• Value iteration: apply the BOE  V(s) ← max_a Σ p(s'|s,a)[r + γV(s')]  until Δ < θ",
        "• Policy iteration: evaluate π exactly (Bellman eq. → linear system), then improve greedily",
        "• Both give the same V* (unit test). Number of sweeps:",
        "      Deterministic grid:  VI 6   ·  PI 6",
        "      Slippery grid:        VI 145  ·  PI 7",
        "      Cliff Walking:         VI 15  ·  PI 15",
        "• VI: cheap sweeps, asymptotic convergence · PI: expensive steps, exact in a few iterations",
    ], Inches(0.7), Inches(1.5), Inches(12), Inches(5.5), size=20)

    s = slide(prs, "Backup: extras beyond the class slides", "Each one is a small change to an algorithm seen in class")
    add_text(s, [
        "• Expected SARSA: target r + γ Σ π(a'|s') Q(s',a')  (Bellman eq. (3) inside the TD target, lower variance)",
        "• Double Q-learning: two tables, one picks argmax, the other evaluates it  (removes the max over-estimation)",
        "• n-step SARSA: n = 1 → SARSA, n → ∞ → Monte Carlo",
        "• Regret = V*(s0) − V^π(s0), computed exactly with the model",
        "• α(s,a) = α/(1+k·N(s,a)): Robbins–Monro; MC's online mean is the case α = 1/N",
        "• Optuna TPE: proposes hyper-parameters where the good trials concentrate; report on held-out seeds",
    ], Inches(0.7), Inches(1.5), Inches(12), Inches(5.5), size=20)

    prs.save(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
