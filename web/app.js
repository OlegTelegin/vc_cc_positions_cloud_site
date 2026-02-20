const DATA_PATH = "./data/1000_positions_jobbert_v2_2d_coords_umap.csv";
const HIGHLIGHT_PATH = "./data/list_by_type_3.csv";
const RIGHT_HIGHLIGHT_PATH = "./data/list_by_type_1.csv";
const MIN_ZOOM = 0.7;
const MAX_ZOOM = 20;
const MARGIN = 28;
const DOT_RADIUS = 2.1;
const NEIGHBOR_K = 4;
const NEIGHBOR_CLASS_THRESHOLD = 3;
const BRIDGE_GROUPS = {
  amir: "Amir's classification from S&M words",
  r2: "All positions from 10-positions-level S&M category",
};

const svg = d3.select("#map");
const tooltip = document.getElementById("tooltip");
const statusEl = document.getElementById("status");
const zoomInBtn = document.getElementById("zoom-in");
const zoomOutBtn = document.getElementById("zoom-out");
const bridgeGroupSelect = document.getElementById("bridge-group-select");
const bridgeListTitleEl = document.getElementById("bridge-list-title");
const bridgePositionListEl = document.getElementById("bridge-position-list");

const scene = svg.append("g").attr("class", "scene");
const pointsLayer = scene.append("g").attr("class", "points-layer");

let width = 0;
let height = 0;
let xScale = null;
let yScale = null;
let zoomBehavior = null;
let currentTransform = d3.zoomIdentity;
let pointsData = [];
let highlightRoleNums = new Set();
let rightHighlightRoleNums = new Set();
let neighborhoodBridgeRoleNums = new Set();
let bridgeSetByGroup = {
  amir: new Set(),
  r2: new Set(),
};
let selectedBridgeGroup = "amir";

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function getMapDimensions() {
  const bounds = svg.node().getBoundingClientRect();
  width = Math.max(320, Math.floor(bounds.width));
  height = Math.max(320, Math.floor(bounds.height));
  svg.attr("viewBox", `0 0 ${width} ${height}`);
}

function buildScales(data) {
  const xExtent = d3.extent(data, (d) => d.x);
  const yExtent = d3.extent(data, (d) => d.y);
  const drawWidth = Math.max(1, width - MARGIN * 2);
  const drawHeight = Math.max(1, height - MARGIN * 2);

  xScale = d3.scaleLinear().domain(xExtent).range([MARGIN, MARGIN + drawWidth]).nice();
  yScale = d3.scaleLinear().domain(yExtent).range([MARGIN + drawHeight, MARGIN]).nice();
}

function positionTooltip(event) {
  const spacing = 14;
  const rect = tooltip.getBoundingClientRect();
  let x = event.clientX + spacing;
  let y = event.clientY + spacing;
  const maxX = window.innerWidth - rect.width - 10;
  const maxY = window.innerHeight - rect.height - 10;

  if (x > maxX) x = Math.max(10, event.clientX - rect.width - spacing);
  if (y > maxY) y = Math.max(10, event.clientY - rect.height - spacing);

  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

function showTooltip(event, d) {
  tooltip.textContent = d.title;
  tooltip.hidden = false;
  positionTooltip(event);
}

function hideTooltip() {
  tooltip.hidden = true;
}

function halfDotPath(cx, cy, radius, side) {
  if (side === "left") {
    return `M ${cx} ${cy - radius} A ${radius} ${radius} 0 0 0 ${cx} ${cy + radius} L ${cx} ${cy - radius} Z`;
  }
  return `M ${cx} ${cy - radius} A ${radius} ${radius} 0 0 1 ${cx} ${cy + radius} L ${cx} ${cy - radius} Z`;
}

function computeNeighborhoodBridgeSet(data, classRoleNums, k = NEIGHBOR_K, minClassNeighbors = NEIGHBOR_CLASS_THRESHOLD) {
  const bridgeSet = new Set();
  if (!data.length || k <= 0) return bridgeSet;

  for (let i = 0; i < data.length; i += 1) {
    const point = data[i];
    if (classRoleNums.has(point.roleNum)) continue;

    const nearest = [];

    for (let j = 0; j < data.length; j += 1) {
      if (i === j) continue;
      const other = data[j];
      const dx = point.x - other.x;
      const dy = point.y - other.y;
      const distSq = dx * dx + dy * dy;
      nearest.push({ distSq, roleNum: other.roleNum });
    }

    nearest.sort((a, b) => a.distSq - b.distSq);

    let classCount = 0;
    const maxNeighbors = Math.min(k, nearest.length);
    for (let n = 0; n < maxNeighbors; n += 1) {
      if (classRoleNums.has(nearest[n].roleNum)) classCount += 1;
    }

    if (classCount >= minClassNeighbors) {
      bridgeSet.add(point.roleNum);
    }
  }

  return bridgeSet;
}

function updateBridgePositionList() {
  if (!bridgePositionListEl || !bridgeListTitleEl) return;

  const groupLabel = BRIDGE_GROUPS[selectedBridgeGroup] || selectedBridgeGroup;
  const ringedPoints = pointsData
    .filter((d) => neighborhoodBridgeRoleNums.has(d.roleNum))
    .sort((a, b) => a.title.localeCompare(b.title));

  bridgeListTitleEl.textContent = `${groupLabel}: positions highlighted in red (${ringedPoints.length})`;

  bridgePositionListEl.innerHTML = "";

  if (!ringedPoints.length) {
    const emptyItem = document.createElement("li");
    emptyItem.textContent = "No positions match this rule.";
    bridgePositionListEl.appendChild(emptyItem);
    return;
  }

  ringedPoints.forEach((point) => {
    const item = document.createElement("li");
    item.textContent = `${point.title} (${point.roleNum})`;
    bridgePositionListEl.appendChild(item);
  });
}

function updateStatusCounts() {
  setStatus(
    `Loaded ${pointsData.length} positions (left: ${highlightRoleNums.size}, right: ${rightHighlightRoleNums.size}, ringed: ${neighborhoodBridgeRoleNums.size}).`
  );
}

function applySelectedBridgeGroup() {
  neighborhoodBridgeRoleNums = bridgeSetByGroup[selectedBridgeGroup] || new Set();
  renderPoints();
  updateBridgePositionList();
  updateStatusCounts();
}

function renderPoints() {
  const dots = pointsLayer
    .selectAll("g.dot")
    .data(pointsData, (d) => d.id)
    .join((enter) => {
      const group = enter.append("g").attr("class", "dot");
      group.append("path").attr("class", "dot-half dot-half-left");
      group.append("path").attr("class", "dot-half dot-half-right");
      group.append("circle").attr("class", "dot-ring");
      return group;
    })
    .classed("is-highlighted", (d) => highlightRoleNums.has(d.roleNum))
    .classed("is-right-highlighted", (d) => rightHighlightRoleNums.has(d.roleNum))
    .classed("is-neighborhood-bridge", (d) => neighborhoodBridgeRoleNums.has(d.roleNum))
    .on("mouseenter", showTooltip)
    .on("mousemove", (event, d) => showTooltip(event, d))
    .on("mouseleave", hideTooltip);

  dots
    .select(".dot-half-left")
    .attr("d", (d) => halfDotPath(xScale(d.x), yScale(d.y), DOT_RADIUS, "left"));

  dots
    .select(".dot-half-right")
    .attr("d", (d) => halfDotPath(xScale(d.x), yScale(d.y), DOT_RADIUS, "right"));

  dots
    .select(".dot-ring")
    .attr("cx", (d) => xScale(d.x))
    .attr("cy", (d) => yScale(d.y))
    .attr("r", DOT_RADIUS + 1.35);
}

function updateTransform(event) {
  currentTransform = event.transform;
  scene.attr("transform", currentTransform.toString());
}

function setupZoom() {
  zoomBehavior = d3
    .zoom()
    .scaleExtent([MIN_ZOOM, MAX_ZOOM])
    .on("start", () => svg.classed("is-dragging", true))
    .on("zoom", updateTransform)
    .on("end", () => svg.classed("is-dragging", false));

  svg.call(zoomBehavior).call(zoomBehavior.transform, currentTransform);

  zoomInBtn.addEventListener("click", () => {
    svg.transition().duration(160).call(zoomBehavior.scaleBy, 1.25);
  });

  zoomOutBtn.addEventListener("click", () => {
    svg.transition().duration(160).call(zoomBehavior.scaleBy, 0.8);
  });
}

function resizeMap() {
  if (!pointsData.length) return;

  getMapDimensions();
  buildScales(pointsData);
  renderPoints();
  currentTransform = d3.zoomIdentity;
  svg.call(zoomBehavior.transform, currentTransform);
}

function setupBridgeControls() {
  if (!bridgeGroupSelect) return;
  bridgeGroupSelect.value = selectedBridgeGroup;
  bridgeGroupSelect.addEventListener("change", () => {
    selectedBridgeGroup = bridgeGroupSelect.value;
    applySelectedBridgeGroup();
  });
}

async function initialize() {
  setStatus("Loading position data...");
  getMapDimensions();

  try {
    const [raw, highlightsRaw, rightHighlightsRaw] = await Promise.all([
      d3.csv(DATA_PATH),
      d3.csv(HIGHLIGHT_PATH),
      d3.csv(RIGHT_HIGHLIGHT_PATH),
    ]);

    highlightRoleNums = new Set(
      highlightsRaw
        .map((row) => Number(row.role_k1000_v3_num))
        .filter((value) => Number.isFinite(value))
    );

    rightHighlightRoleNums = new Set(
      rightHighlightsRaw
        .map((row) => Number(row.role_k1000_v3_num))
        .filter((value) => Number.isFinite(value))
    );

    pointsData = raw
      .map((row, idx) => ({
        id: idx,
        title: row.role_k1000_v3?.trim() || "(untitled position)",
        roleNum: Number(row.role_k1000_v3_num),
        x: Number(row.x_2d),
        y: Number(row.y_2d),
      }))
      .filter((d) => Number.isFinite(d.x) && Number.isFinite(d.y));

    if (!pointsData.length) {
      throw new Error("No usable points found in CSV.");
    }

    bridgeSetByGroup = {
      amir: computeNeighborhoodBridgeSet(pointsData, highlightRoleNums),
      r2: computeNeighborhoodBridgeSet(pointsData, rightHighlightRoleNums),
    };

    buildScales(pointsData);
    applySelectedBridgeGroup();
    setupZoom();
  } catch (error) {
    console.error(error);
    setStatus("Failed to load data. Check CSV path and file permissions.", true);
  }
}

setupBridgeControls();
window.addEventListener("resize", resizeMap);
window.addEventListener("blur", hideTooltip);

initialize();
