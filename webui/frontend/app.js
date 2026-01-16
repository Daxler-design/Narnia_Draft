const $ = (id) => document.getElementById(id);

const apiBaseInput = $("apiBase");
const statusText = $("statusText");

const profilePathInput = $("profilePath");
const bracingPathInput = $("bracingPath");
const generateBracingInput = $("generateBracing");
const numCentroidsInput = $("numCentroids");
const ridgeSigmaInput = $("ridgeSigma");

const postprocessEnableInput = $("postprocessEnable");
const closeRadiusInput = $("closeRadius");
const minAreaInput = $("minArea");
const temporalWindowInput = $("temporalWindow");
const profileOffsetInput = $("profileOffset");
const bracingOffsetInput = $("bracingOffset");

const booleanModeInput = $("booleanMode");
const booleanProfileOffsetInput = $("booleanProfileOffset");
const booleanBracingOffsetInput = $("booleanBracingOffset");
const resultIsoInput = $("resultIso");

const datasetIdInput = $("datasetId");
const sliceIndexInput = $("sliceIndex");
const isoThresholdInput = $("isoThreshold");
const sliceCountEl = $("sliceCount");
const zCoordEl = $("zCoord");
const curveCountsEl = $("curveCounts");

const computeBtn = $("computeBtn");
const postprocessBtn = $("postprocessBtn");
const booleanBtn = $("booleanBtn");
const previewBtn = $("previewBtn");

const fieldCanvas = $("fieldCanvas");
const curveSvg = $("curveSvg");

const state = {
  datasetId: "",
};

const getBase = () => {
  const base = apiBaseInput.value.trim();
  return base ? base.replace(/\/$/, "") : "";
};

const apiUrl = (path) => `${getBase()}${path}`;

const setStatus = (msg) => {
  statusText.textContent = msg;
};

const setDatasetId = (datasetId) => {
  state.datasetId = datasetId || "";
  datasetIdInput.value = state.datasetId;
};

const jsonHeaders = { "Content-Type": "application/json" };

const postJson = async (path, payload) => {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
};

const getJson = async (path) => {
  const response = await fetch(apiUrl(path));
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
};

const handleCompute = async () => {
  setStatus("Running static bracing...");
  const payload = {
    profile_json_path: profilePathInput.value.trim(),
    bracing_json_path: bracingPathInput.value.trim() || null,
    generate_bracing: generateBracingInput.checked,
    params: {
      num_centroids: Number(numCentroidsInput.value),
      ridge_sigma: Number(ridgeSigmaInput.value),
    },
  };

  try {
    const resp = await postJson("/compute/static-bracing", payload);
    setDatasetId(resp.dataset_id);
    setStatus(resp.message || "Static bracing complete.");
  } catch (err) {
    setStatus(`Compute failed: ${err.message}`);
  }
};

const handlePostprocess = async () => {
  const datasetId = datasetIdInput.value.trim();
  if (!datasetId) {
    setStatus("Dataset ID required for postprocess.");
    return;
  }
  setStatus("Applying postprocess...");
  const payload = {
    enable: postprocessEnableInput.checked,
    close_radius: Number(closeRadiusInput.value),
    min_area: Number(minAreaInput.value),
    temporal_window: Number(temporalWindowInput.value),
    profile_offset: Number(profileOffsetInput.value),
    bracing_offset: Number(bracingOffsetInput.value),
    dataset_id: datasetId,
  };

  try {
    const resp = await postJson("/compute/postprocess", payload);
    setStatus(resp.message || "Postprocess complete.");
  } catch (err) {
    setStatus(`Postprocess failed: ${err.message}`);
  }
};

const handleBoolean = async () => {
  const datasetId = datasetIdInput.value.trim();
  if (!datasetId) {
    setStatus("Dataset ID required for boolean.");
    return;
  }
  setStatus("Applying boolean...");
  const payload = {
    mode: booleanModeInput.value,
    profile_offset: Number(booleanProfileOffsetInput.value),
    bracing_offset: Number(booleanBracingOffsetInput.value),
    result_iso: Number(resultIsoInput.value),
    dataset_id: datasetId,
  };

  try {
    const resp = await postJson("/compute/boolean", payload);
    setStatus(resp.message || "Boolean complete.");
  } catch (err) {
    setStatus(`Boolean failed: ${err.message}`);
  }
};

const resizeCanvas = () => {
  const rect = fieldCanvas.parentElement.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(320, Math.floor(rect.height));
  fieldCanvas.width = width;
  fieldCanvas.height = height;
  curveSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
};

const drawField = (field2d) => {
  if (!field2d || !field2d.length) {
    return;
  }
  const ny = field2d.length;
  const nx = field2d[0].length;
  const values = field2d.flat();
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (const v of values) {
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }
  const range = maxVal - minVal || 1;

  const offscreen = document.createElement("canvas");
  offscreen.width = nx;
  offscreen.height = ny;
  const ctx = offscreen.getContext("2d");
  const image = ctx.createImageData(nx, ny);
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const v = field2d[y][x];
      const t = (v - minVal) / range;
      const shade = Math.max(0, Math.min(255, Math.round(t * 255)));
      const idx = (y * nx + x) * 4;
      image.data[idx] = shade;
      image.data[idx + 1] = shade;
      image.data[idx + 2] = shade;
      image.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);

  const dest = fieldCanvas.getContext("2d");
  dest.clearRect(0, 0, fieldCanvas.width, fieldCanvas.height);
  dest.drawImage(offscreen, 0, 0, fieldCanvas.width, fieldCanvas.height);
};

const toPath = (curve, minX, maxX, minY, maxY, width, height) => {
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  return curve
    .map((point, index) => {
      const x = ((point[0] - minX) / spanX) * width;
      const y = height - ((point[1] - minY) / spanY) * height;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
};

const drawCurves = (curves, xCoords, yCoords) => {
  curveSvg.innerHTML = "";
  if (!curves || !xCoords || !yCoords) {
    return;
  }
  const width = fieldCanvas.width;
  const height = fieldCanvas.height;

  const minX = Math.min(...xCoords);
  const maxX = Math.max(...xCoords);
  const minY = Math.min(...yCoords);
  const maxY = Math.max(...yCoords);

  const addCurves = (list, className) => {
    list.forEach((curve) => {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", toPath(curve, minX, maxX, minY, maxY, width, height));
      path.setAttribute("class", className);
      curveSvg.appendChild(path);
    });
  };

  addCurves(curves.result || [], "curve-result");
  addCurves(curves.profile || [], "curve-profile");
  addCurves(curves.bracing || [], "curve-bracing");
};

const handlePreview = async () => {
  const datasetId = datasetIdInput.value.trim();
  if (!datasetId) {
    setStatus("Dataset ID required for preview.");
    return;
  }
  const sliceIndex = Number(sliceIndexInput.value);
  const isoThreshold = Number(isoThresholdInput.value);
  const query = `/result/preview?dataset_id=${encodeURIComponent(datasetId)}&slice_index=${sliceIndex}&iso_threshold=${isoThreshold}`;

  try {
    setStatus("Loading preview...");
    const preview = await getJson(query);
    setDatasetId(preview.dataset_id);
    sliceCountEl.textContent = preview.slice_count ?? "-";
    zCoordEl.textContent = preview.z_coord?.toFixed(3) ?? "-";
    const counts = [
      preview.curves?.result?.length ?? 0,
      preview.curves?.profile?.length ?? 0,
      preview.curves?.bracing?.length ?? 0,
    ];
    curveCountsEl.textContent = `R ${counts[0]} / P ${counts[1]} / B ${counts[2]}`;

    sliceIndexInput.max = Math.max(0, (preview.slice_count || 1) - 1);
    resizeCanvas();
    drawField(preview.fields?.result);
    drawCurves(preview.curves, preview.x_coords, preview.y_coords);
    setStatus("Preview loaded.");
  } catch (err) {
    setStatus(`Preview failed: ${err.message}`);
  }
};

generateBracingInput.addEventListener("change", () => {
  bracingPathInput.disabled = generateBracingInput.checked;
});

computeBtn.addEventListener("click", handleCompute);
postprocessBtn.addEventListener("click", handlePostprocess);
booleanBtn.addEventListener("click", handleBoolean);
previewBtn.addEventListener("click", handlePreview);

window.addEventListener("resize", () => {
  resizeCanvas();
});

resizeCanvas();
setStatus("Ready.");
