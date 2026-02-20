const DATA_PATH = "./data/1000_positions_jobbert_v2_2d_coords_umap.csv";
const CLASSIFICATION_SOURCES_PATH = "./data/classification_sources.csv";
const RANKINGS_PATH = "./data/classification_r2_rankings.csv";
const REGRESSION_STRUCTURE_PATH = "./data/regression_structure.csv";
const MIN_ZOOM = 0.7;
const MAX_ZOOM = 20;
const MARGIN = 28;
const DOT_RADIUS = 2.1;
const NEIGHBOR_K = 4;
const NEIGHBOR_CLASS_THRESHOLD = 3;

const svg = d3.select("#map");
const tooltip = document.getElementById("tooltip");
const statusEl = document.getElementById("status");
const zoomInBtn = document.getElementById("zoom-in");
const zoomOutBtn = document.getElementById("zoom-out");
const leftClassSelect = document.getElementById("map-left-select");
const rightClassSelect = document.getElementById("map-right-select");
const bridgeGroupSelect = document.getElementById("bridge-group-select");
const bridgePositionListEl = document.getElementById("bridge-position-list");
const comparisonChartEl = document.getElementById("comparison-chart");

const scene = svg.append("g").attr("class", "scene");
const pointsLayer = scene.append("g").attr("class", "points-layer");

let width = 0;
let height = 0;
let xScale = null;
let yScale = null;
let zoomBehavior = null;
let currentTransform = d3.zoomIdentity;
let pointsData = [];
let leftRoleNums = new Set();
let rightRoleNums = new Set();
let neighborhoodBridgeRoleNums = new Set();
let classificationOptions = [];
let roleSetByFileName = {};
let rankingByRegressionFile = {};
let regressionStructureRows = [];
let selectedLeftFile = "";
let selectedRightFile = "";
let selectedBridgeGroup = "left";

function getDisplayTitleByFileName(fileName) {
  return classificationOptions.find((option) => option.fileName === fileName)?.displayTitle || fileName;
}

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
  if (!bridgePositionListEl) return;

  const activeLabel = selectedBridgeGroup === "right"
    ? getDisplayTitleByFileName(selectedRightFile)
    : getDisplayTitleByFileName(selectedLeftFile);
  const ringedPoints = pointsData
    .filter((d) => neighborhoodBridgeRoleNums.has(d.roleNum))
    .sort((a, b) => a.title.localeCompare(b.title));

  bridgePositionListEl.innerHTML = "";

  if (!ringedPoints.length) {
    const emptyItem = document.createElement("li");
    emptyItem.textContent = `No positions match this rule for "${activeLabel}".`;
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
    `Loaded ${pointsData.length} positions (left: ${leftRoleNums.size}, right: ${rightRoleNums.size}, ringed: ${neighborhoodBridgeRoleNums.size}).`
  );
}

function getRankLabel(rank) {
  if (!Number.isFinite(rank)) return "n/a";
  const mod100 = rank % 100;
  if (mod100 >= 11 && mod100 <= 13) return `${rank}th`;
  const mod10 = rank % 10;
  if (mod10 === 1) return `${rank}st`;
  if (mod10 === 2) return `${rank}nd`;
  if (mod10 === 3) return `${rank}rd`;
  return `${rank}th`;
}

function formatCoefficient(value) {
  if (!Number.isFinite(value)) return "n/a";
  if (value > 1000 || value < -1000) return `${(value / 1000).toFixed(1)}K`;
  if (value > 10 || value < -10) return `${Math.round(value)}`;
  return `${value.toFixed(1)}`;
}

function rankToRelativeHeight(rank, maxRank = 4) {
  const boundedRank = Number.isFinite(rank) ? Math.min(Math.max(rank, 1), maxRank) : maxRank;
  if (maxRank <= 1) return 1;
  const step = 0.6 / (maxRank - 1);
  return 1 - (boundedRank - 1) * step;
}

function normalizeWithWhat(value) {
  if (!value) return "";
  if (value.includes("employment")) return "employment";
  if (value.includes("salary")) return "salaries";
  if (value.includes("total_compensation")) return "total_compensation";
  return value;
}

function getRegressionBySlot(sampleRestrict, whatFirms, predict, withWhat) {
  return regressionStructureRows.find(
    (row) =>
      row.sample_restrict === sampleRestrict &&
      row.what_firms === whatFirms &&
      row.predict === predict &&
      normalizeWithWhat(row.with_what) === withWhat
  );
}

function renderComparisonChart() {
  if (!comparisonChartEl) return;
  const chart = d3.select(comparisonChartEl);
  const bounds = comparisonChartEl.getBoundingClientRect();
  const width = Math.max(420, Math.floor(bounds.width));
  const height = Math.max(340, Math.floor(bounds.height));
  chart.attr("viewBox", `0 0 ${width} ${height}`);
  chart.selectAll("*").remove();

  const margins = { top: 1, right: 1, bottom: 1, left: 1 };
  const innerWidth = Math.max(1, width - margins.left - margins.right);
  const innerHeight = Math.max(1, height - margins.top - margins.bottom);

  const sampleColumns = [
    { key: "restricted", title: "Sample, where Revelio size is close to CIQ size" },
    { key: "full", title: "Full Sample" },
  ];
  const rowSlots = [
    { whatFirms: "all", predict: "spending", title: "All Firms - Predict spending" },
    { whatFirms: "all", predict: "spending_per_employee", title: "All Firms - Predict spending per employee" },
    { whatFirms: "smallest", predict: "spending", title: "Smallest firms (1st quartile) - Predict spending" },
    { whatFirms: "smallest", predict: "spending_per_employee", title: "Smallest firms (1st quartile) - Predict spending per employee" },
  ];
  const withWhatOrder = ["employment", "salaries", "total_compensation"];
  const withWhatLabel = {
    employment: "Employment",
    salaries: "Salaries",
    total_compensation: "Total Compensation",
  };

  const colGap = 4;
  const colWidth = (innerWidth - colGap) / 2;
  const rowGap = 2;
  const rowHeight = (innerHeight - 11 - rowGap * 3) / 4;
  const sectionTop = margins.top + 10;

  sampleColumns.forEach((sampleCol, colIdx) => {
    const colX = margins.left + colIdx * (colWidth + colGap);

    chart
      .append("text")
      .attr("x", colX + colWidth / 2)
        .attr("y", margins.top + 7)
      .attr("text-anchor", "middle")
        .attr("font-size", "9")
      .attr("font-weight", "700")
      .attr("fill", "#ccd6f6")
      .text(sampleCol.title);

    rowSlots.forEach((slot, rowIdx) => {
      const boxY = sectionTop + rowIdx * (rowHeight + rowGap);

      chart
        .append("rect")
        .attr("x", colX)
        .attr("y", boxY)
        .attr("width", colWidth)
        .attr("height", rowHeight)
        .attr("rx", 6)
        .attr("fill", "rgba(10, 25, 47, 0.28)")
        .attr("stroke", "rgba(136, 146, 176, 0.35)");

      chart
        .append("text")
        .attr("x", colX + 8)
        .attr("y", boxY + 9)
        .attr("font-size", "8")
        .attr("fill", "#8892b0")
        .text(slot.title);

      const pairGap = 6;
      const pairWidth = (colWidth - 16 - pairGap * 2) / 3;
      const pairTop = boxY + 10;
      const pairBottom = boxY + rowHeight - 8;
      const y = d3.scaleLinear().domain([0, 1]).range([pairBottom, pairTop + 6]);

      chart
        .append("text")
        .attr("x", colX + 4)
        .attr("y", pairBottom + 7)
        .attr("font-size", "7")
        .attr("fill", "#8892b0")
        .text("Coef.");

      withWhatOrder.forEach((withWhat, pairIdx) => {
        const pairX = colX + 8 + pairIdx * (pairWidth + pairGap);
        const regDef = getRegressionBySlot(sampleCol.key, slot.whatFirms, slot.predict, withWhat);
        if (!regDef) return;

        chart
          .append("text")
          .attr("x", pairX + pairWidth / 2)
          .attr("y", pairTop + 1)
          .attr("text-anchor", "middle")
          .attr("font-size", "7")
          .attr("fill", "#ccd6f6")
          .text(withWhatLabel[withWhat]);

        const barGap = 2;
        const barWidth = (pairWidth - barGap) / 2;
        const regNum = Number(regDef.regression_number);
        const leftKey = `${regNum}|${selectedLeftFile}`;
        const rightKey = `${regNum}|${selectedRightFile}`;
        const leftData = rankingByRegressionFile[leftKey] || { r2: 0, rank: NaN, coefficient: NaN };
        const rightData = rankingByRegressionFile[rightKey] || { r2: 0, rank: NaN, coefficient: NaN };
        const pairData = [
          { side: "left", ...leftData, x: pairX, heightScore: rankToRelativeHeight(leftData.rank) },
          { side: "right", ...rightData, x: pairX + barWidth + barGap, heightScore: rankToRelativeHeight(rightData.rank) },
        ];

        pairData.forEach((bar) => {
          const barTop = y(bar.heightScore);
          const barHeight = Math.max(1, pairBottom - barTop);
          chart
            .append("rect")
            .attr("x", bar.x)
            .attr("y", barTop)
            .attr("width", barWidth)
            .attr("height", barHeight)
            .attr("rx", 3)
            .attr("fill", bar.side === "left" ? "rgba(255, 176, 95, 0.95)" : "rgba(128, 170, 255, 0.95)");

          chart
            .append("text")
            .attr("x", bar.x + barWidth / 2)
            .attr("y", barTop + 10)
            .attr("text-anchor", "middle")
            .attr("font-size", "14")
            .attr("font-weight", "700")
            .attr("fill", "#081a2f")
            .text(getRankLabel(bar.rank));

          chart
            .append("text")
            .attr("x", bar.x + barWidth / 2)
            .attr("y", barTop + 24)
            .attr("text-anchor", "middle")
            .attr("font-size", "10")
            .attr("fill", "#0f2741")
            .text(`R²=${bar.r2.toFixed(2)}`);
        });

        chart
          .append("text")
          .attr("x", pairX + barWidth / 2)
          .attr("y", pairBottom + 7)
          .attr("text-anchor", "middle")
          .attr("font-size", "7")
          .attr("fill", "#ccd6f6")
          .text(formatCoefficient(leftData.coefficient));

        chart
          .append("text")
          .attr("x", pairX + barWidth + barGap + barWidth / 2)
          .attr("y", pairBottom + 7)
          .attr("text-anchor", "middle")
          .attr("font-size", "7")
          .attr("fill", "#ccd6f6")
          .text(formatCoefficient(rightData.coefficient));
      });
    });
  });
}

function applySelectedClassifications() {
  leftRoleNums = roleSetByFileName[selectedLeftFile] || new Set();
  rightRoleNums = roleSetByFileName[selectedRightFile] || new Set();
  const bridgeReferenceSet = selectedBridgeGroup === "right" ? rightRoleNums : leftRoleNums;
  neighborhoodBridgeRoleNums = computeNeighborhoodBridgeSet(pointsData, bridgeReferenceSet);
  renderPoints();
  renderComparisonChart();
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
    .classed("is-highlighted", (d) => leftRoleNums.has(d.roleNum))
    .classed("is-right-highlighted", (d) => rightRoleNums.has(d.roleNum))
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
  renderComparisonChart();
  currentTransform = d3.zoomIdentity;
  svg.call(zoomBehavior.transform, currentTransform);
}

function populateClassificationSelect(selectEl, selectedFileName) {
  if (!selectEl) return;
  selectEl.innerHTML = "";
  classificationOptions.forEach((option) => {
    const opt = document.createElement("option");
    opt.value = option.fileName;
    opt.textContent = option.displayTitle;
    if (option.fileName === selectedFileName) opt.selected = true;
    selectEl.appendChild(opt);
  });
}

function populateBridgeGroupSelect() {
  if (!bridgeGroupSelect) return;
  bridgeGroupSelect.innerHTML = "";

  const leftOption = document.createElement("option");
  leftOption.value = "left";
  leftOption.textContent = getDisplayTitleByFileName(selectedLeftFile);
  bridgeGroupSelect.appendChild(leftOption);

  const rightOption = document.createElement("option");
  rightOption.value = "right";
  rightOption.textContent = getDisplayTitleByFileName(selectedRightFile);
  bridgeGroupSelect.appendChild(rightOption);

  bridgeGroupSelect.value = selectedBridgeGroup;
}

function ensureDistinctSelections(changedSide) {
  if (selectedLeftFile !== selectedRightFile || classificationOptions.length < 2) return;
  const fallback = classificationOptions.find((option) => option.fileName !== selectedLeftFile)?.fileName;
  if (!fallback) return;
  if (changedSide === "left") {
    selectedRightFile = fallback;
    if (rightClassSelect) rightClassSelect.value = selectedRightFile;
  } else {
    selectedLeftFile = fallback;
    if (leftClassSelect) leftClassSelect.value = selectedLeftFile;
  }
}

function setupClassificationControls() {
  if (leftClassSelect) {
    leftClassSelect.addEventListener("change", () => {
      selectedLeftFile = leftClassSelect.value;
      ensureDistinctSelections("left");
      populateBridgeGroupSelect();
      applySelectedClassifications();
    });
  }

  if (rightClassSelect) {
    rightClassSelect.addEventListener("change", () => {
      selectedRightFile = rightClassSelect.value;
      ensureDistinctSelections("right");
      populateBridgeGroupSelect();
      applySelectedClassifications();
    });
  }

  if (bridgeGroupSelect) {
    bridgeGroupSelect.addEventListener("change", () => {
      selectedBridgeGroup = bridgeGroupSelect.value;
      applySelectedClassifications();
    });
  }
}

async function initialize() {
  setStatus("Loading position data...");
  getMapDimensions();

  try {
    const [raw, classificationSources, rankingsRaw, regressionStructureRaw] = await Promise.all([
      d3.csv(DATA_PATH),
      d3.csv(CLASSIFICATION_SOURCES_PATH),
      d3.csv(RANKINGS_PATH),
      d3.csv(REGRESSION_STRUCTURE_PATH),
    ]);

    classificationOptions = classificationSources
      .map((row) => ({
        fileName: row.file_name?.trim(),
        displayTitle: row.display_title?.trim(),
      }))
      .filter((row) => row.fileName && row.displayTitle);

    if (!classificationOptions.length) {
      throw new Error("No classification options found.");
    }

    const uniqueFileNames = Array.from(new Set(classificationOptions.map((option) => option.fileName)));
    const classificationFileRows = await Promise.all(
      uniqueFileNames.map((fileName) => d3.csv(`./data/${fileName}.csv`))
    );

    roleSetByFileName = {};
    uniqueFileNames.forEach((fileName, index) => {
      roleSetByFileName[fileName] = new Set(
        classificationFileRows[index]
          .map((row) => Number(row.role_k1000_v3_num))
          .filter((value) => Number.isFinite(value))
      );
    });

    rankingByRegressionFile = {};
    rankingsRaw.forEach((row) => {
      const fileName = row.file_name?.trim();
      const regressionNumber = Number(row.regression_number);
      if (!fileName || !Number.isFinite(regressionNumber)) return;
      rankingByRegressionFile[`${regressionNumber}|${fileName}`] = {
        r2: Number(row.r2),
        rank: Number(row.rank),
        coefficient: Number(row.coefficient),
      };
    });

    regressionStructureRows = regressionStructureRaw
      .map((row) => ({
        regression_number: Number(row.regression_number),
        predict: row.predict?.trim(),
        with_what: row.with_what?.trim(),
        sample_restrict: row.sample_restrict?.trim(),
        what_firms: row.what_firms?.trim(),
      }))
      .filter((row) => Number.isFinite(row.regression_number));

    selectedLeftFile = classificationOptions[0].fileName;
    selectedRightFile = classificationOptions[Math.min(1, classificationOptions.length - 1)].fileName;
    ensureDistinctSelections("right");
    populateClassificationSelect(leftClassSelect, selectedLeftFile);
    populateClassificationSelect(rightClassSelect, selectedRightFile);
    populateBridgeGroupSelect();

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

    buildScales(pointsData);
    applySelectedClassifications();
    setupZoom();
  } catch (error) {
    console.error(error);
    setStatus("Failed to load data. Check CSV path and file permissions.", true);
  }
}

setupClassificationControls();
window.addEventListener("resize", resizeMap);
window.addEventListener("blur", hideTooltip);

initialize();
