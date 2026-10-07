// Builds docs/presentation.pptx: a 5-minute talk plus backup slides for the questions.
//
//   cd docs/presentation && npm install && node build.js
//
// Every number on the slides comes from results/tables/. The figures come from results/figures/.

const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa6");

const ROOT = path.resolve(__dirname, "..", "..");
const FIG = (name) => path.join(ROOT, "results", "figures", name);
const OUT = path.join(ROOT, "docs", "presentation.pptx");

// ---------------------------------------------------------------------------------------------
// Look and feel. The accents are the same colours the algorithms have in every figure, so a
// colour means the same thing on a slide and in a plot.
const THEME = {
  name: "Gridworld",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0F1B2D", // navy: text, dark slides
    lt1: "FFFFFF",
    dk2: "4A5568", // muted text
    lt2: "EEF2F6", // soft panels
    accent1: "D55E00", // Q-learning
    accent2: "0072B2", // SARSA
    accent3: "009E73", // Expected SARSA / goal
    accent4: "E69F00", // Double Q / the agent
    accent5: "56B4E9", // n-step SARSA
    accent6: "7A7A7A", // Monte Carlo
    hlink: "0072B2",
    folHlink: "56B4E9",
  },
};
const HEX = THEME.colors;
const NAVY_CELL = "1C2C45"; // grid cells on the dark slides
const W = 13.333, H = 7.5;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Tabular Reinforcement Learning";
pres.author = "Igor Ruiz, Andoni Garrido";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------------------------------------------------------------------------------------------
// Layouts
const FOOTER = { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.text2, margin: 0 };

pres.defineSlideMaster({
  title: "TITLE",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.0, w: 6.9, h: 1.7, fontSize: 46, bold: true, color: C.background1, align: "left", valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.85, w: 6.9, h: 1.1, fontSize: 20, color: "CADCFC", valign: "top", margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.8, fontSize: 32, bold: true, color: C.text1, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Tabular RL · Assignment 1", options: { ...FOOTER } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: HEX.dk2, align: "right" },
});

pres.defineSlideMaster({
  title: "DARK",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.8, fontSize: 32, bold: true, color: C.background1, valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: "8A9BB5", align: "right" },
});

// ---------------------------------------------------------------------------------------------
// Small drawing helpers

async function icon(Icon, color = "FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { color: "#" + color, size: String(size) }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

// A rounded "grid cell": the visual motif of the deck.
function cell(slide, x, y, size, fill, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w: size, h: size, rectRadius: size * 0.18, fill: { color: fill }, line: { color: fill }, objectName: name,
  });
}

// A cell with a number or a short label in it (used as a list marker).
function badge(slide, x, y, size, fill, label, fontSize = 16) {
  cell(slide, x, y, size, fill, `badge ${label}`);
  slide.addText(label, { x, y, w: size, h: size, align: "center", valign: "middle", fontSize, bold: true, color: "FFFFFF", margin: 0, isTextBox: true });
}

// An icon centred in a coloured cell.
function iconCell(slide, x, y, size, fill, data, name) {
  cell(slide, x, y, size, fill, name);
  const pad = size * 0.24;
  slide.addImage({ data, x: x + pad, y: y + pad, w: size - 2 * pad, h: size - 2 * pad, altText: name });
}

// The 3x4 class gridworld. `dark` = drawn on a navy slide.
function drawGridworld(slide, x0, y0, s, gap, { dark = false, path = false } = {}) {
  const free = dark ? NAVY_CELL : HEX.lt2;
  const special = { 3: [HEX.accent3, "+1"], 7: [HEX.accent1, "−1"], 5: [dark ? "33445E" : "C5CEDA", ""] };
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 4; c++) {
      const st = 4 * r + c;
      const [fill, label] = special[st] || [free, ""];
      const x = x0 + c * (s + gap), y = y0 + r * (s + gap);
      cell(slide, x, y, s, fill, `grid cell ${st}`);
      if (label) slide.addText(label, { x, y, w: s, h: s, align: "center", valign: "middle", fontSize: Math.round(s * 22), bold: true, color: "FFFFFF", margin: 0, isTextBox: true });
    }
  }
  const centre = (r, c) => [x0 + c * (s + gap) + s / 2, y0 + r * (s + gap) + s / 2];
  if (path) {
    // the optimal path: up, up, right, right, right
    const pts = [[2, 0], [1, 0], [0, 0], [0, 1], [0, 2], [0, 3]].map(([r, c]) => centre(r, c));
    pts[pts.length - 1][0] = x0 + 3 * (s + gap) - 0.06; // stop at the goal's edge, not on its label
    for (let i = 0; i < pts.length - 1; i++) {
      const [ax, ay] = pts[i], [bx, by] = pts[i + 1];
      slide.addShape(pres.shapes.LINE, {
        x: Math.min(ax, bx), y: Math.min(ay, by), w: Math.abs(bx - ax) || 0.001, h: Math.abs(by - ay) || 0.001,
        flipV: by < ay, line: { color: dark ? "FFFFFF" : HEX.dk1, width: 2.25, dashType: "dash", endArrowType: i === pts.length - 2 ? "triangle" : undefined },
      });
    }
  }
  // the agent
  const [ax, ay] = centre(2, 0);
  slide.addShape(pres.shapes.OVAL, { x: ax - s * 0.27, y: ay - s * 0.27, w: s * 0.54, h: s * 0.54, fill: { color: HEX.accent4 }, line: { color: HEX.accent4 }, objectName: "agent" });
}

// Cliff Walking, 4x12, small.
function drawCliff(slide, x0, y0, s, gap) {
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 12; c++) {
      let fill = HEX.lt2;
      if (r === 3 && c > 0 && c < 11) fill = HEX.accent1;
      if (r === 3 && c === 11) fill = HEX.accent3;
      if (r === 3 && c === 0) fill = HEX.accent4;
      cell(slide, x0 + c * (s + gap), y0 + r * (s + gap), s, fill, `cliff cell ${r},${c}`);
    }
  }
}

// A big number with a caption under it.
function stat(slide, x, y, w, value, caption, color) {
  slide.addText(value, { x, y, w, h: 0.8, fontSize: 40, bold: true, color, margin: 0, valign: "bottom", isTextBox: true });
  slide.addText(caption, { x, y: y + 0.85, w, h: 0.7, fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true });
}

// A soft panel (no edge stripes): a tinted rounded rectangle.
function panel(slide, x, y, w, h, fill = HEX.lt2, name = "panel") {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: name });
}

function image(slide, file, x, y, w, ratio, alt) {
  slide.addImage({ path: FIG(file), x, y, w, h: w / ratio, altText: alt });
}

const CHART_TEXT = { catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, legendFontSize: 12, legendColor: HEX.dk2, titleColor: HEX.dk1, titleFontSize: 14 };

// pptxgenjs always writes Office's default palette into the theme, so "scheme" colours (titles,
// text, accents) would come out Office blue. This swaps our palette into ppt/theme/*.xml.
async function writeThemeColors(file, theme) {
  const fs = require("fs");
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const order = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];
  const scheme = `<a:clrScheme name="${theme.name}">` +
    order.map((k) => `<a:${k}><a:srgbClr val="${theme.colors[k]}"/></a:${k}>`).join("") + "</a:clrScheme>";
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/theme\/theme\d+\.xml$/.test(n))) {
    const xml = await zip.file(name).async("string");
    zip.file(name, xml.replace(/<a:clrScheme[\s\S]*?<\/a:clrScheme>/, scheme));
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

// ---------------------------------------------------------------------------------------------
async function build() {
  const ic = {
    dice: await icon(fa.FaDice), target: await icon(fa.FaBullseye), robot: await icon(fa.FaRobot),
    shield: await icon(fa.FaShieldHalved), bolt: await icon(fa.FaBolt), compass: await icon(fa.FaCompass),
    steps: await icon(fa.FaShoePrints), eraser: await icon(fa.FaEraser), check: await icon(fa.FaCircleCheck),
  };

  pres.addSection({ title: "Talk" });

  // 1 · Title -------------------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Talk" });
    s.addText("Tabular Reinforcement Learning", { placeholder: "title" });
    s.addText("SARSA vs Q-learning, and what it takes to make them converge", { placeholder: "body" });
    s.addText("Igor Ruiz  ·  Andoni Garrido", { x: 0.7, y: 5.6, w: 6.9, h: 0.4, fontSize: 16, bold: true, color: "FFFFFF", margin: 0, isTextBox: true });
    s.addText("Reinforcement Learning · Universidad de Deusto · October 2026", { x: 0.7, y: 6.0, w: 6.9, h: 0.4, fontSize: 13, color: "8A9BB5", margin: 0, isTextBox: true });
    drawGridworld(s, 8.25, 1.95, 1.0, 0.14, { dark: true, path: true });
    s.addText("start", { x: 8.25, y: 5.38, w: 1.0, h: 0.3, fontSize: 11, color: "8A9BB5", align: "center", margin: 0, isTextBox: true });
    s.addNotes("(~15 s) We implemented SARSA and Q-learning for the class gridworld, deterministic and slippery, and then asked: why do they behave differently, and what makes them converge? On the right is the class grid: the agent starts bottom-left and the dashed line is the optimal path to +1, avoiding the -1 pit.");
  }

  // 2 · Setup --------------------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
    s.addText("Three environments, one honest yardstick", { placeholder: "title" });

    // left: the environments
    const rows = [
      ["Gridworld · deterministic", "The 3×4 grid from class: +1 goal, −1 pit, one wall"],
      ["Gridworld · slippery", "80% the intended move, 10% to each side"],
      ["Cliff Walking", "−1 per step, −100 for falling off the cliff"],
    ];
    drawGridworld(s, 0.6, 1.55, 0.3, 0.05, {});
    drawGridworld(s, 0.6, 3.25, 0.3, 0.05, {});
    drawCliff(s, 0.6, 4.95, 0.115, 0.02);
    rows.forEach(([name, desc], i) => {
      const y = 1.55 + i * 1.7;
      s.addText(name, { x: 2.35, y, w: 4.2, h: 0.45, fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true });
      s.addText(desc, { x: 2.35, y: y + 0.45, w: 4.2, h: 0.6, fontSize: 15, color: C.text2, margin: 0, valign: "top", isTextBox: true });
    });

    // right: how we score
    panel(s, 7.0, 1.45, 5.7, 5.15);
    s.addText("How we score every run", { x: 7.4, y: 1.7, w: 5.0, h: 0.5, fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true });
    const how = [
      [ic.robot, HEX.accent2, "Model-free agents", "they only call reset() and step()"],
      [ic.target, HEX.accent1, "Exact regret", "V*(s0) − V^π(s0), with Q* from value iteration"],
      [ic.dice, HEX.accent3, "20 seeds each", "report spread, not one lucky run"],
    ];
    how.forEach(([data, color, head, text], i) => {
      const y = 2.5 + i * 1.3;
      iconCell(s, 7.4, y, 0.75, color, data, head);
      s.addText(head, { x: 8.4, y: y - 0.02, w: 4.0, h: 0.4, fontSize: 17, bold: true, color: C.text1, margin: 0, isTextBox: true });
      s.addText(text, { x: 8.4, y: y + 0.38, w: 4.0, h: 0.45, fontSize: 14, color: C.text2, margin: 0, isTextBox: true });
    });
    s.addNotes("(~35 s) Three environments: the class grid, the same grid with slippery 80/10/10 moves, and Cliff Walking from Sutton & Barto. The agents never see the transition model, they only play. But we do know the model, so we compute the true optimum with value iteration and grade every learned policy exactly: regret is how much value the greedy policy loses from the start. Every configuration is trained with 20 seeds.");
  }

  // 3 · Minimum requirement -------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
    s.addText("Minimum requirement: both solve the class gridworld", { placeholder: "title" });
    image(s, "exp1_values_policies.png", 0.6, 1.4, 8.6, 1.73, "Learned values and greedy policies of SARSA and Q-learning next to the optimal Q*, deterministic and slippery gridworld");
    stat(s, 9.7, 1.35, 3.0, "20 / 20", "seeds optimal for both on the deterministic grid", HEX.accent3);
    stat(s, 9.7, 3.15, 3.0, "19 vs 15", "Q-learning vs SARSA seeds optimal, slippery grid, default settings", HEX.accent1);
    stat(s, 9.7, 4.95, 3.0, "20 / 20", "both, once ε → 0 and α shrinks with visits", HEX.accent2);
    s.addNotes("(~50 s) Left: the true optimal values and policy, then what SARSA and Q-learning learned. Deterministic: both find the optimal path in all 20 seeds. Slippery: notice the optimum changes, next to the pit it is better to walk away from it. With default settings Q-learning gets 19 of 20 seeds right but SARSA only 15: SARSA learns the value of its own exploring policy, so its values near the pit are lower. When exploration fades to zero and the step size shrinks with the visits, the theory's convergence conditions, both get 20 of 20.");
  }

  // 4 · Cliff ------------------------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
    s.addText("On-policy vs off-policy: the cliff", { placeholder: "title" });
    image(s, "exp3_cliff_walking.png", 0.6, 1.4, 8.6, 1.75, "SARSA learns a safe path, Q-learning the edge path; online return and regret over training");
    const cards = [
      [HEX.accent2, ic.shield, "SARSA", ["Safe path · 17 steps", "Return while exploring −25.8", "Greedy policy optimal 0 / 20"]],
      [HEX.accent1, ic.bolt, "Q-learning", ["Edge path · 13 steps", "Return while exploring −51.9", "Greedy policy optimal 20 / 20"]],
    ];
    cards.forEach(([color, data, name, lines], i) => {
      const y = 1.4 + i * 2.35;
      panel(s, 9.6, y, 3.1, 2.1);
      iconCell(s, 9.85, y + 0.25, 0.55, color, data, name);
      s.addText(name, { x: 10.55, y: y + 0.25, w: 2.0, h: 0.55, fontSize: 19, bold: true, color, valign: "middle", margin: 0, isTextBox: true });
      s.addText(lines.map((t, k) => ({ text: t, options: { breakLine: k < lines.length - 1 } })), { x: 9.85, y: y + 0.95, w: 2.75, h: 1.0, fontSize: 14, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 3, isTextBox: true });
    });
    s.addText("SARSA learns the policy it follows, random steps included. Q-learning learns the greedy one.", { x: 9.6, y: 6.15, w: 3.1, h: 0.75, fontSize: 13, italic: true, color: C.text2, margin: 0, valign: "top", isTextBox: true });
    s.addNotes("(~45 s) The classic Cliff Walking result. SARSA's target uses the action it really takes next, exploration included, so it keeps a row of distance from the cliff. Q-learning's target is the max, it learns the shortest path along the edge, which is optimal, but while it is still exploring it falls more often: worse return during training, yet its greedy policy is optimal in all 20 seeds.");
  }

  // 5 · All algorithms -------------------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
    s.addText("Six algorithms, three environments", { placeholder: "title" });
    const envs = ["Deterministic grid", "Slippery grid", "Cliff Walking"];
    const series = [
      ["Monte Carlo", [80.6, 72.8, 41.2]],
      ["SARSA", [81.1, 85.0, 56.4]],
      ["n-step SARSA", [80.0, 76.1, 59.1]],
      ["Expected SARSA", [82.2, 87.2, 74.7]],
      ["Q-learning", [94.4, 97.8, 95.4]],
      ["Double Q-learning", [95.0, 88.9, 67.8]],
    ];
    s.addChart(pres.charts.BAR, series.map(([name, values]) => ({ name, labels: envs, values })), {
      x: 0.6, y: 1.35, w: 8.2, h: 5.4, barDir: "col", barGapWidthPct: 60,
      chartColors: [HEX.accent6, HEX.accent2, HEX.accent5, HEX.accent3, HEX.accent1, HEX.accent4],
      showTitle: true, title: "States where the final greedy action is optimal (%)",
      valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20,
      valGridLine: { color: "E3E8EF", size: 0.75 }, catGridLine: { style: "none" },
      showLegend: true, legendPos: "b", ...CHART_TEXT,
    });
    const notes = [
      [HEX.accent1, "1", "Q-learning is the most accurate everywhere", "≥ 94% optimal actions in all three"],
      [HEX.accent3, "2", "Expected SARSA is the best on-policy method", "averaging over a' removes noise"],
      [HEX.accent6, "3", "Monte Carlo fails in Cliff Walking", "it needs episodes that end, and they don't"],
    ];
    notes.forEach(([color, n, head, text], i) => {
      const y = 1.6 + i * 1.65;
      badge(s, 9.2, y, 0.6, color, n, 18);
      s.addText(head, { x: 10.0, y: y - 0.05, w: 2.75, h: 0.75, fontSize: 16, bold: true, color: C.text1, margin: 0, valign: "top", isTextBox: true });
      s.addText(text, { x: 10.0, y: y + 0.72, w: 2.75, h: 0.6, fontSize: 13, color: C.text2, margin: 0, valign: "top", isTextBox: true });
    });
    s.addNotes("(~35 s) We compared six tabular methods with the same settings. Q-learning is the most accurate in all three environments. Among the on-policy methods, Expected SARSA is the best: it averages over the next action instead of sampling it, so its updates are less noisy. Monte Carlo collapses in Cliff Walking: it learns from complete episodes, and with a bad early policy the episodes almost never end.");
  }

  // 6 · Optuna -----------------------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
    s.addText("Optuna: tuning helps every algorithm", { placeholder: "title" });
    stat(s, 0.6, 1.25, 3.8, "12 / 12", "studies learn significantly faster (p < 0.05, 20 held-out seeds)", HEX.accent1);
    stat(s, 4.8, 1.25, 3.8, "0 → 18", "seeds where Expected SARSA finds the optimal cliff path", HEX.accent3);
    stat(s, 9.0, 1.25, 3.7, "8 / 10", "TD settings that picked a shrinking α (Robbins–Monro) on their own", HEX.accent2);
    const algos = ["SARSA", "Exp. SARSA", "Q-learning", "Double Q", "n-step", "MC const-α"];
    const charts = [
      ["Slippery grid · mean regret (log)", [0.0689, 0.0343, 0.0101, 0.0729, 0.1829, 0.4004], [0.0396, 0.0197, 0.0028, 0.0158, 0.0319, 0.0455]],
      ["Cliff Walking · mean regret (log)", [2.3933, 0.8601, 0.5895, 3.5263, 2.6814, 2.7989], [0.8955, 0.5447, 0.0865, 0.6735, 0.5342, 1.8235]],
    ];
    charts.forEach(([title, def, tuned], i) => {
      s.addChart(pres.charts.BAR, [
        { name: "default", labels: algos, values: def },
        { name: "tuned", labels: algos, values: tuned },
      ], {
        x: 0.6 + i * 6.15, y: 3.15, w: 5.95, h: 3.65, barDir: "col", barGapWidthPct: 50,
        chartColors: ["B8C2D0", HEX.accent1], showTitle: true, title,
        valAxisLogScaleBase: 10, valGridLine: { color: "E3E8EF", size: 0.75 }, catGridLine: { style: "none" },
        valAxisLabelFormatCode: "0.###", showLegend: true, legendPos: "t", ...CHART_TEXT, catAxisLabelFontSize: 11,
      });
    });
    s.addNotes("(~50 s) We tuned all six algorithms in both hard environments with Optuna: 12 studies, 150 trials each, with two goals at once, learn fast and end up exactly optimal. Then we retrained default and tuned settings on 20 new seeds. All 12 learn significantly faster. Expected SARSA goes from never finding the optimal cliff path to 18 of 20 seeds. And the nice part: in 8 of 10 cases Optuna picked a step size that shrinks with the visits, and exploration that fades to zero. It rediscovered the convergence conditions from theory.");
  }

  // 7 · Training lessons ------------------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
    s.addText("What makes training work", { placeholder: "title" });
    const cards = [
      [HEX.accent3, ic.compass, "Explore, then stop", "On-policy methods only reach the optimum when ε → 0 (GLIE)."],
      [HEX.accent2, ic.steps, "Shrink the steps", "α / (1 + k·N(s,a)) lets values settle even with random moves."],
      [HEX.accent1, ic.eraser, "Forget bad early returns", "MC with 1/N got stuck in 14/20 seeds at −100. A constant α: 0/20."],
    ];
    cards.forEach(([color, data, head, text], i) => {
      const x = 0.6 + i * 4.1;
      panel(s, x, 1.4, 3.9, 2.05);
      iconCell(s, x + 0.3, 1.65, 0.6, color, data, head);
      s.addText(head, { x: x + 1.05, y: 1.65, w: 2.7, h: 0.6, fontSize: 17, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
      s.addText(text, { x: x + 0.3, y: 2.4, w: 3.35, h: 0.95, fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true });
    });
    image(s, "exp6_failure_analysis.png", 1.87, 3.7, 9.6, 3.0, "Per-seed returns of the variants in Cliff Walking, and Double Q estimates collapsing to the -100 plateau");
    s.addNotes("(~40 s) Three lessons about training. Exploration must fade out for on-policy methods to become optimal. The step size should shrink with the visits so the values can settle despite random moves. And the cliff taught us the last one: a policy that never reaches the goal is worth minus 100. Monte Carlo averaging every return ever seen keeps the terrible first episodes forever and gets stuck there in 14 of 20 seeds; with a constant step size it forgets them and none gets stuck. The red lines on the right are Double Q seeds stuck at that same minus 100.");
  }

  // 8 · Takeaways -------------------------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Talk" });
    s.addText("Takeaways", { placeholder: "title" });
    const items = [
      [HEX.accent3, "SARSA and Q-learning both solve the class gridworld, deterministic and slippery"],
      [HEX.accent1, "Q-learning learns the optimum; SARSA learns the policy it follows, so it plays safe"],
      [HEX.accent2, "Convergence needs ε → 0 and a shrinking α, and Optuna found exactly that"],
      [HEX.accent4, "Fully reproducible: cached results, 9 tests, one command redraws everything"],
    ];
    items.forEach(([color, text], i) => {
      const y = 1.55 + i * 1.15;
      badge(s, 0.6, y, 0.75, color, String(i + 1), 22);
      s.addText(text, { x: 1.65, y, w: 6.6, h: 0.75, fontSize: 19, color: "FFFFFF", valign: "middle", margin: 0, isTextBox: true });
    });
    drawGridworld(s, 9.0, 1.7, 0.85, 0.12, { dark: true, path: true });
    s.addText("Questions?", { x: 9.0, y: 4.85, w: 3.7, h: 0.6, fontSize: 26, bold: true, color: "FFFFFF", align: "center", margin: 0, isTextBox: true });
    s.addText("github.com/ElTiToRuiz/assignment-1", { x: 9.0, y: 5.45, w: 3.7, h: 0.4, fontSize: 13, color: "8A9BB5", align: "center", margin: 0, isTextBox: true });
    s.addNotes("(~20 s) To sum up: both algorithms solve the gridworld; Q-learning learns the optimum while SARSA learns the policy it actually follows and plays safe; convergence needs exploration and step sizes that fade, which is what Optuna chose; and everything is reproducible from the repository. Thank you, questions?");
  }

  // Backup ------------------------------------------------------------------------------------------
  pres.addSection({ title: "Backup" });
  {
    const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Backup" });
    s.addText("Backup", { placeholder: "title" });
    s.addText("Extra detail for the questions", { x: 0.6, y: 1.2, w: 12.1, h: 0.5, fontSize: 18, color: "CADCFC", align: "center", margin: 0, isTextBox: true });
    drawGridworld(s, (W - (4 * 0.85 + 3 * 0.12)) / 2, 2.6, 0.85, 0.12, { dark: true });
  }
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
    s.addText("Dynamic programming: the answer key", { placeholder: "title" });
    const head = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.dk1 }, align: "center" } });
    s.addTable([
      [head("Environment"), head("Value iteration (sweeps)"), head("Policy iteration (steps)")],
      ["Deterministic grid", "6", "6"],
      ["Slippery grid", "145", "7"],
      ["Cliff Walking", "15", "15"],
    ].map((row, i) => i === 0 ? row : row.map((t, j) => ({ text: t, options: { align: j ? "center" : "left", fill: { color: i % 2 ? HEX.lt2 : "FFFFFF" } } }))),
    { x: 0.6, y: 1.5, w: 7.2, colW: [2.6, 2.3, 2.3], fontSize: 16, color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "D5DCE6" }, rowH: 0.55 });
    const pts = [
      ["Value iteration", "applies the Bellman optimality equation until V stops changing: cheap sweeps, many of them"],
      ["Policy iteration", "solves the Bellman equation exactly, then improves greedily: few, expensive steps"],
      ["Same V*", "a unit test checks both agree; agents never see P"],
    ];
    pts.forEach(([h, t], i) => {
      const y = 1.5 + i * 1.35;
      badge(s, 8.4, y, 0.5, [HEX.accent2, HEX.accent1, HEX.accent3][i], String(i + 1), 16);
      s.addText(h, { x: 9.1, y: y - 0.04, w: 3.6, h: 0.4, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
      s.addText(t, { x: 9.1, y: y + 0.36, w: 3.6, h: 0.85, fontSize: 13, color: C.text2, margin: 0, valign: "top", isTextBox: true });
    });
  }
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
    s.addText("Convergence conditions in the slippery grid", { placeholder: "title" });
    image(s, "exp1_convergent_schedule.png", 0.6, 1.6, 12.1, 3.63, "Default schedule vs GLIE plus Robbins-Monro: share of seeds exactly optimal, regret and error over episodes");
    s.addText("Dashed: default (ε ≥ 0.05, constant α). Solid: ε → 0 and α = 0.5 / (1 + 0.005·N). Both reach 20/20 seeds; SARSA needs about 15× more episodes than Q-learning.", { x: 0.6, y: 5.25, w: 12.1, h: 0.8, fontSize: 15, color: C.text2, margin: 0, valign: "top", isTextBox: true });
  }
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
    s.addText("Monte Carlo vs TD, and how much to explore", { placeholder: "title" });
    image(s, "exp5c_bias_variance.png", 0.6, 1.35, 6.0, 2.97, "Bias and variance of Q(s0,a*) for Monte Carlo, SARSA and Q-learning");
    image(s, "exp5a_alpha_sensitivity.png", 6.75, 1.35, 6.0, 2.97, "Learning-rate sensitivity of SARSA and Q-learning");
    image(s, "exp5b_exploration.png", 1.2, 3.55, 11.0, 3.71, "Q-learning with different exploration schedules");
  }
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
    s.addText("Optuna: Pareto fronts and what matters", { placeholder: "title" });
    image(s, "exp4_pareto.png", 1.15, 1.25, 11.0, 3.06, "Every Optuna trial on the speed and exactness plane, with Pareto front, default and chosen configuration");
    image(s, "exp4_param_importance.png", 0.6, 4.95, 4.4, 2.41, "Parameter importance per algorithm and environment");
    s.addText("Two goals at once: learn fast and end exactly optimal. We keep the Pareto front and pick the point with the smallest sum. Trial 0 is always the default (square).", { x: 5.4, y: 5.15, w: 7.3, h: 1.2, fontSize: 15, color: C.text2, margin: 0, valign: "top", isTextBox: true });
  }

  await pres.writeFile({ fileName: OUT });
  await writeThemeColors(OUT, THEME);
  console.log("saved", OUT);
}

build().catch((e) => { console.error(e); process.exit(1); });
