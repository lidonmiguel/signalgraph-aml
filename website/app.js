"use strict";

// All numbers come from a versioned, aggregate-only file deployed with the page.
const DATA_URL = "data/public-results.json";
const SVG_NS = "http://www.w3.org/2000/svg";
const formatCount = new Intl.NumberFormat("en-US");
const budgetButtons = [...document.querySelectorAll("[data-capacity]")];
const budgetRange = document.getElementById("budget-range");
let published = null;

function svgElement(name, attributes, label = "") {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  if (label) element.textContent = label;
  element.classList.add("plot");
  return element;
}

function chart(data, selectedIndex) {
  const svg = document.getElementById("capacity-chart");
  svg.querySelectorAll(".plot").forEach((element) => element.remove());
  const x = (index) => 68 + index * 145;
  const y = (hits) => 220 - hits * 2.5;

  for (const value of [0, 20, 40, 60]) {
    const position = y(value);
    svg.append(svgElement("line", {x1: 56, y1: position, x2: 666, y2: position, stroke: "#dce8df", "stroke-width": 1}));
    svg.append(svgElement("text", {x: 35, y: position + 4, "text-anchor": "end"}, String(value)));
  }
  svg.append(svgElement("rect", {x: x(selectedIndex) - 20, y: 40, width: 40, height: 196, fill: "#e5f5e9", rx: 7}));
  for (const [values, color] of [[data.historical.hits, "#e7a272"], [data.current.hits, "#1d9573"]]) {
    const points = values.map((hits, index) => `${x(index)},${y(hits)}`).join(" ");
    svg.append(svgElement("polyline", {points, fill: "none", stroke: color, "stroke-width": 3, "stroke-linecap": "round", "stroke-linejoin": "round"}));
    values.forEach((hits, index) => {
      svg.append(svgElement("circle", {cx: x(index), cy: y(hits), r: index === selectedIndex ? 6 : 4, fill: color, stroke: "white", "stroke-width": 2}));
    });
  }
  data.capacities.forEach((capacity, index) => {
    svg.append(svgElement("text", {x: x(index), y: 253, "text-anchor": "middle"}, formatCount.format(capacity)));
  });
}

function renderMethods(data, index) {
  const list = document.getElementById("method-list");
  list.replaceChildren();
  const maxHits = Math.max(...data.comparison.map((method) => method.hits[index]));
  for (const method of data.comparison) {
    const row = document.createElement("div");
    row.className = "method-row";
    const title = document.createElement("div");
    const name = document.createElement("span");
    name.className = "method-label";
    name.textContent = method.label;
    const training = document.createElement("span");
    training.className = "method-training";
    training.textContent = method.training;
    title.append(name, training);

    const track = document.createElement("div");
    track.className = "method-bar";
    track.setAttribute("role", "img");
    track.setAttribute("aria-label", `${method.label}: ${method.hits[index]} positives`);
    const bar = document.createElement("span");
    bar.style.width = `${100 * method.hits[index] / maxHits}%`;
    track.append(bar);
    const count = document.createElement("strong");
    count.className = "method-hits";
    count.textContent = String(method.hits[index]);
    const auc = document.createElement("span");
    auc.className = "method-auc";
    auc.textContent = method.pr_auc.toFixed(5);
    auc.title = `PR-AUC; ${method.seconds} seconds per method, excluding shared setup`;
    row.append(title, track, count, auc);
    list.append(row);
  }
  const footer = document.createElement("div");
  footer.className = "method-footer";
  const note = document.createElement("span");
  note.textContent = "Bars = hits · right column = PR-AUC";
  const time = document.createElement("span");
  time.textContent = "Timings exclude shared setup";
  footer.append(note, time);
  list.append(footer);
}

function selectBudget(index) {
  if (!published || index < 0 || index >= published.capacities.length) return;
  const capacity = published.capacities[index];
  const hits = published.current.hits[index];
  const precision = hits / capacity;
  const recall = hits / published.dataset.evaluation_positives;
  const prevalence = published.dataset.evaluation_positives / published.dataset.evaluation_cases;
  document.getElementById("budget-label").textContent = `${formatCount.format(capacity)} cases`;
  document.getElementById("metric-hits").textContent = String(hits);
  document.getElementById("metric-hits-note").textContent = `of ${formatCount.format(capacity)} cases reviewed`;
  document.getElementById("metric-precision").textContent = `${(100 * precision).toFixed(1)}%`;
  document.getElementById("metric-recall").textContent = `${(100 * recall).toFixed(2)}%`;
  document.getElementById("metric-lift").textContent = `${(precision / prevalence).toFixed(2)}×`;
  document.getElementById("compare-budget").textContent = `K = ${formatCount.format(capacity)}`;
  budgetRange.value = String(index);
  budgetButtons.forEach((button) => button.setAttribute("aria-pressed", String(Number(button.dataset.capacity) === capacity)));
  document.querySelectorAll("[data-row-capacity]").forEach((row) => row.classList.toggle("selected", Number(row.dataset.rowCapacity) === capacity));
  chart(published, index);
  renderMethods(published, index);
}

budgetRange.addEventListener("input", () => selectBudget(Number(budgetRange.value)));
budgetButtons.forEach((button) => button.addEventListener("click", () => selectBudget(published?.capacities.indexOf(Number(button.dataset.capacity)) ?? -1)));

fetch(DATA_URL)
  .then((response) => {
    if (!response.ok) throw new Error("Public data unavailable");
    return response.json();
  })
  .then((data) => {
    if (data.schema_version !== 1 || data.capacities.length !== 5) throw new Error("Unexpected data version");
    published = data;
    selectBudget(1);
    const review = data.drift_review;
    document.getElementById("train-share").textContent = `${(100 * review.training_any / data.dataset.training_cases).toFixed(2)}%`;
    document.getElementById("later-share").textContent = `${(100 * review.later_any / data.dataset.evaluation_cases).toFixed(2)}%`;
    document.getElementById("train-bar").style.width = `${100 * review.training_any / data.dataset.training_cases}%`;
    document.getElementById("later-bar").style.width = `${100 * review.later_any / data.dataset.evaluation_cases}%`;
    document.getElementById("data-status").textContent = "Versioned aggregate results loaded.";
  })
  .catch(() => {
    document.getElementById("method-list").textContent = "Comparison chart unavailable. Open the source report in the evidence section.";
    document.getElementById("data-status").textContent = "Aggregate data could not be loaded. Static figures and source links remain available.";
    budgetRange.disabled = true;
    budgetButtons.forEach((button) => { button.disabled = true; });
  });
