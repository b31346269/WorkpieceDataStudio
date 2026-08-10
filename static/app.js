const $ = (id) => document.getElementById(id);
const state = {
  system: null,
  projects: [],
  project: null,
  candidates: [],
  filtered: [],
  candidateIndex: 0,
  image: new Image(),
  boxes: [],
  selectedBoxId: null,
  drag: null,
  modelUploadController: null,
};

const COLORS = ["#55d6be", "#ffb45b", "#70a8ff", "#d98cff", "#ff6978"];

const OVERHEAD_FACTORY_BASE = `Machine-vision dataset photograph captured by an unseen high-mounted camera outside the frame. Request an 88 to 90 degree camera elevation to compensate for the generator's tendency to make the apparent angle too low, producing an apparent output angle of about 80 to 90 degrees. Keep the optical axis almost perpendicular to the work surface. Keep a realistic camera-to-workpiece working distance of 70 to 90 cm, preferably about 80 cm, using a normal smartphone or inspection-camera lens rather than an ultra-wide lens. The horizontal work surface and local metal fixture fill the entire background edge-to-edge. There is no distant background and nothing hangs above the workpiece. The complete workpiece occupies only about 20 to 30 percent of the image area, with broad visible fixture and work-surface margin on all four sides. The top face remains the dominant visible surface and appears close to its true plan-view shape with nearly parallel opposite edges. Side walls must occupy less than 10 percent of the visible workpiece height.`;

const FACTORY_REALISM = `Use realistic aluminum texture, machining marks, scratches, grease, dust, oxidation, localized overhead point light, contact shadows, mild smartphone noise and imperfect exposure. Documentary factory inspection photography, photorealistic and physically plausible, not CGI, not CAD and not a clean studio product photograph.`;

const FASTENER_REALISM = `Use varied but mechanically authentic industrial screws and bolts. Every visible fastener intended as a screw must have a clearly recessed drive cavity in the center of its top face; this requirement also applies to hexagonal outer heads. Each generated workpiece must visibly contain both empty holes and installed screws: approximately 8 to 18 mechanically plausible drilled or threaded holes and a sparse, non-dense arrangement of approximately 2 to 4 clearly visible screws, scaled naturally to the housing size. The exact count may vary between images; prioritize broad empty spacing and never cluster screw heads. Never satisfy the screw requirement with empty threaded holes. Use one or two drive styles within one image and rotate the drive-style selection across the batch. Use a varied mix of deep internal-hex sockets, deep six-lobe Torx sockets, recessed Phillips crosses and recessed straight slots. Vary the outer profile among cylindrical socket-cap, button-head, countersunk, flange and hexagonal combination heads, but never use a conventional blank-top external hex bolt. The recess must have visible depth, crisp edges and a contact shadow: a shallow engraved circle, concentric ring, tiny center dot or painted symbol is not a drive recess. Do not place any smooth raised cylindrical boss, plug, cap, post or dowel anywhere on the workpiece because it can be confused with a screw. Every raised cylindrical feature must instead be either an open bore or bearing seat with a clearly visible deep opening, or a fastener with one of the required recessed drives. Never create a headless threaded stud, protruding threaded rod, exposed set screw or bare male thread on the workpiece: every installed fastener must terminate in a clearly visible recessed-drive head. Keep each fastener straight, correctly scaled and seated flush against a machined surface or washer, with a small realistic contact shadow. Do not fuse screws into the casting and do not create floating, melted or decorative fasteners.`;

const OVERHEAD_COMPOSITION = `Composition is mandatory: request an 88 to 90 degree elevation so the apparent result remains within 80 to 90 degrees, with the top surface facing the viewer and side walls reduced to a very thin rim. No low oblique angle, no three-quarter product view, no horizon, no strong vanishing point, no ceiling, no walls, no shelves, no distant machinery, and no camera, spindle, probe, lamp or inspection head anywhere inside the frame.`;

const WORKPIECE_RECIPES = {
  square_center: `Create a clearly different but mechanically plausible low-profile boxy square or short-rectangular cast-aluminum gearbox housing. Keep the housing shallow so its side walls remain a thin rim in the overhead image. Include one substantial circular bearing seat, machined flange or shaft interface near the center of the top face, occupying roughly 25 to 45 percent of its width. Use a varied flange diameter and bolt pattern, four-corner or irregular mounting ears, rectilinear or radial reinforcement ribs, recessed cast cavities, cooling fins, small threaded holes and correctly seated fasteners. Change the silhouette, proportions and detailed arrangement between images; do not copy one fixed central-flange design.`,
  offset_flange: `Create a clearly different but mechanically plausible cast-aluminum gearbox housing with one dominant large bearing seat, flange or shaft interface placed decisively near a randomly selected side or corner, at least one flange radius away from the geometric center. The geometric center must remain uninterrupted cast metal, ribs or a rectangular recessed panel; never put a circular opening at the center. Use an asymmetric external silhouette, unequal mounting ears, strongly non-mirrored reinforcement ribs, offset recessed cavities, varied cooling fins, circular threaded holes and correctly seated fasteners. Keep the structure functional and manufacturable while avoiding bilateral symmetry and repeated layouts.`,
  no_center_circle: `Create a clearly different but mechanically plausible cast-aluminum gearbox housing with no large circular feature near the center. Make rectangular or irregular machined covers, recessed panels, rib grids, cooling-fin groups, stepped cast surfaces or offset shaft interfaces the dominant top-face features. Circular geometry is limited to smaller drilled holes, bearing seats and fasteners away from the center. Vary the silhouette, proportions, mounting ears, cavities and rib arrangement while preserving manufacturable construction.`,
  scale_light_fixture: `Create a mechanically plausible cast-aluminum gearbox housing with strongly varied physical proportions and framing while strictly preserving the apparent 80 to 90 degree overhead viewpoint. Alternate among compact square, elongated rectangular and irregular housings; vary the complete workpiece occupancy from about 18 to 32 percent of the image and rotate it naturally on the fixture without increasing camera obliqueness. Keep broad background margin on all four sides. Use believable conveyor pallets, clamping rails, locating pins or worn metal inspection plates. Vary dim ambient light, localized point light from outside the frame, mixed cool factory light, partial underexposure, hard contact shadows and metallic highlights while keeping holes and fasteners clearly reviewable.`,
};

const GENERATION_NEGATIVE = "smooth round metal cap presented as a screw, featureless circular fastener head, blank-top external hex bolt, smooth machined bolt head, shallow engraved ring instead of a drive recess, circular outline without a cavity, center dot instead of a drive recess, smooth raised cylindrical boss, smooth vertical post, blank cylindrical cap, raised dowel, all holes and no installed screws, zero visible screws, more than four installed screws, screw-covered workpiece, dense fastener pattern, clustered screws, adjacent screw heads, headless threaded stud, protruding threaded rod, bare male thread above the workpiece, set screw without a visible drive head, cylindrical plug mistaken for a screw, rivet mistaken for a screw, missing drive recess, off-center drive recess, malformed hex socket, melted screw, fused fastener, floating fastener, camera closer than 70 cm, camera farther than 90 cm, close-up, macro view, cropped workpiece, workpiece filling more than one third of the frame, insufficient background margin, ultra-wide lens, fisheye distortion, apparent camera elevation below 80 degrees, low angle, eye level, strongly oblique camera, three-quarter product photo, tall housing, deep box, large visible front face, large visible side face, side wall taller than 10 percent of workpiece, side-dominant composition, strong trapezoid perspective, horizon, strong vanishing point, ceiling, walls, shelves, distant factory interior, visible camera, hanging camera head, spindle, probe, lamp, inspection head";

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response;
}

function toast(message, error = false) {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast${error ? " error" : ""}`;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, 4500);
}

async function initialize() {
  bindEvents();
  try {
    state.system = await api("/api/system");
    renderSystem();
    await loadProjects();
  } catch (error) {
    toast(error.message, true);
  }
}

function bindEvents() {
  $("refreshProjects").onclick = loadProjects;
  $("createProject").onclick = createProject;
  $("deleteProject").onclick = deleteCurrentProject;
  $("projectSelect").onchange = () => selectProject($("projectSelect").value);
  $("uploadDataset").onclick = uploadDataset;
  $("uploadReference").onclick = uploadReference;
  $("uploadModel").onclick = uploadModel;
  $("clearModelFile").onclick = clearModelFile;
  $("modelFile").onchange = () => {
    $("clearModelFile").disabled = !$("modelFile").files.length;
  };
  $("startGeneration").onclick = startGeneration;
  $("reloadCandidates").onclick = loadCandidates;
  $("candidateFilter").onchange = filterCandidates;
  $("previousCandidate").onclick = () => navigateCandidate(-1);
  $("nextCandidate").onclick = () => navigateCandidate(1);
  $("deleteBox").onclick = deleteSelectedBox;
  $("approveCandidate").onclick = () => saveCandidate("approved");
  $("savePending").onclick = () => saveCandidate("pending");
  $("rejectCandidate").onclick = () => saveCandidate("rejected");
  $("exportDataset").onclick = exportDataset;
  $("annotationClass").onchange = changeSelectedClass;
  $("strength").oninput = () => $("strengthValue").textContent = $("strength").value;
  $("adapterScale").oninput = () => $("adapterValue").textContent = $("adapterScale").value;
  $("qualityMode").onchange = applyQualityPreset;
  $("provider").onchange = applyQualityPreset;
  $("scenePreset").onchange = applyScenePreset;
  $("workpieceRecipe").onchange = applyWorkpieceRecipe;
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.onclick = () => showTab(button.dataset.tab);
  });
  const canvas = $("annotationCanvas");
  canvas.addEventListener("mousedown", canvasDown);
  canvas.addEventListener("mousemove", canvasMove);
  canvas.addEventListener("mouseup", canvasUp);
  canvas.addEventListener("mouseleave", canvasUp);
  window.addEventListener("keydown", (event) => {
    if (event.key === "Delete" && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
      deleteSelectedBox();
    }
  });
}

function renderSystem() {
  const badge = $("systemBadge");
  if (state.system.ml_ready && state.system.cuda) {
    badge.textContent = `AI 就緒 · ${state.system.gpu}`;
    badge.className = "badge ready";
    $("provider").value = (state.system.vram_gb || 0) >= 20
      ? "flux2_klein"
      : "diffusers";
    applyQualityPreset();
  } else {
    badge.textContent = "核心模式 · 尚未安裝本機 AI";
    badge.className = "badge warn";
    $("provider").value = "mock";
  }
}

async function loadProjects() {
  state.projects = await api("/api/projects");
  const select = $("projectSelect");
  const previous = state.project?.id || select.value;
  select.innerHTML = '<option value="">請選擇</option>' + state.projects
    .map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`)
    .join("");
  const desired = state.projects.some((item) => item.id === previous)
    ? previous
    : state.projects[0]?.id;
  if (desired) {
    select.value = desired;
    await selectProject(desired);
  } else {
    state.project = null;
    $("workspace").hidden = true;
    $("noProject").hidden = false;
  }
}

async function createProject() {
  const name = $("newProjectName").value.trim();
  if (!name) return toast("請輸入專案名稱。", true);
  try {
    const project = await api("/api/projects", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name, classes: ["hole", "screw", "tool"]}),
    });
    await loadProjects();
    $("projectSelect").value = project.id;
    await selectProject(project.id);
    toast("專案已建立。");
  } catch (error) { toast(error.message, true); }
}

async function deleteCurrentProject() {
  const project = state.project;
  if (!project) return;
  const confirmed = window.confirm(
    `確定刪除專案「${project.name}」？\n\n專案內的參考圖、候選圖片、模型與匯出檔都會一併刪除，無法復原。`,
  );
  if (!confirmed) return;
  try {
    await api(`/api/projects/${project.id}`, {method: "DELETE"});
    state.project = null;
    state.candidates = [];
    await loadProjects();
    toast(`專案「${project.name}」已刪除。`);
  } catch (error) {
    toast(error.message, true);
  }
}

async function selectProject(projectId) {
  if (!projectId) return;
  state.project = await api(`/api/projects/${projectId}`);
  $("workspace").hidden = false;
  $("noProject").hidden = true;
  renderProject();
  await loadCandidates();
}

function renderProject() {
  const project = state.project;
  const source = project.source_dataset;
  $("datasetSummary").textContent = source
    ? `檔案：${source.filename}\n圖片：${source.image_count}\n標註檔：${source.label_count}\n物件：${source.annotation_count}\nClass IDs：${source.class_ids.join(", ")}`
    : "尚未匯入";
  $("modelList").innerHTML = project.models.length
    ? project.models.map((model) => `
      <div class="item item-row">
        <span>${escapeHtml(model.original_name)} · ${(model.size / 1048576).toFixed(1)} MB</span>
        <button class="danger secondary small-button" data-delete-model="${escapeHtml(model.id)}">移除</button>
      </div>`).join("")
    : '<div class="muted">尚未上傳模型</div>';
  $("modelList").querySelectorAll("[data-delete-model]").forEach((button) => {
    button.onclick = () => deleteUploadedModel(button.dataset.deleteModel);
  });
  $("referenceGrid").innerHTML = project.references.length
    ? project.references.map((ref) => `
      <div class="reference-card">
        <img src="/api/projects/${project.id}/references/${ref.id}/image" alt="">
        <div title="${escapeHtml(ref.original_name)}">${escapeHtml(ref.original_name)}</div>
      </div>`).join("")
    : '<div class="muted">尚未加入參考圖</div>';

  $("generationReference").innerHTML = project.references
    .map((ref) => `<option value="${ref.id}">${escapeHtml(ref.original_name)}</option>`)
    .join("");
  $("generationModel").innerHTML = '<option value="">不使用預標模型</option>' + project.models
    .map((model) => `<option value="${model.id}">${escapeHtml(model.original_name)}</option>`)
    .join("");
  $("annotationClass").innerHTML = project.classes
    .map((name, index) => `<option value="${index}">${index} · ${escapeHtml(name)}</option>`)
    .join("");
  renderCounts();
}

async function uploadFile(inputId, endpoint, formName, signal = undefined) {
  const file = $(inputId).files[0];
  if (!file) throw new Error("請先選擇檔案。");
  const form = new FormData();
  form.append(formName, file);
  return api(endpoint, {method: "POST", body: form, signal});
}

async function uploadDataset() {
  try {
    await uploadFile("datasetFile", `/api/projects/${state.project.id}/dataset`, "dataset");
    await refreshProject();
    toast("Roboflow ZIP 檢查完成。");
  } catch (error) { toast(error.message, true); }
}

async function uploadReference() {
  try {
    await uploadFile("referenceFile", `/api/projects/${state.project.id}/references`, "image");
    await refreshProject();
    toast("參考圖已加入。");
  } catch (error) { toast(error.message, true); }
}

async function uploadModel() {
  if (state.modelUploadController) {
    state.modelUploadController.abort();
    return;
  }
  if (!$("modelFile").files.length) return toast("請先選擇 .pt 檔案。", true);
  const controller = new AbortController();
  state.modelUploadController = controller;
  $("uploadModel").textContent = "取消上傳";
  $("uploadModel").classList.add("danger");
  $("clearModelFile").disabled = true;
  try {
    const result = await uploadFile(
      "modelFile",
      `/api/projects/${state.project.id}/models`,
      "model",
      controller.signal,
    );
    clearModelFile();
    await refreshProject();
    toast(result.duplicate ? "這個模型已存在，沒有重複儲存。" : "YOLO 模型已上傳。");
  } catch (error) {
    if (error.name === "AbortError") {
      toast("已取消模型上傳。");
      await refreshProject();
    } else {
      toast(error.message, true);
    }
  } finally {
    state.modelUploadController = null;
    $("uploadModel").textContent = "上傳 .pt";
    $("uploadModel").classList.remove("danger");
    $("clearModelFile").disabled = !$("modelFile").files.length;
  }
}

function clearModelFile() {
  $("modelFile").value = "";
  $("clearModelFile").disabled = true;
}

async function deleteUploadedModel(modelId) {
  const model = state.project.models.find((item) => item.id === modelId);
  if (!model) return;
  if (!window.confirm(`移除模型「${model.original_name}」？`)) return;
  try {
    await api(`/api/projects/${state.project.id}/models/${modelId}`, {method: "DELETE"});
    await refreshProject();
    toast("模型已移除。");
  } catch (error) {
    toast(error.message, true);
  }
}

function applyQualityPreset() {
  const presets = {
    strict: {
      strength: 0.34,
      adapter: 0.82,
      steps: 32,
      hint: "嚴格模式主要改變光線、材質與背景，並限制結構變形；任何不圓的孔洞、融合或漂浮螺絲仍應直接淘汰。",
    },
    shape_variation: {
      strength: 0.64,
      adapter: 0.60,
      steps: 40,
      hint: "形狀變體會明顯改變工件外輪廓、比例、散熱鰭片與安裝耳；ControlNet 只在前段保護機械邊緣，請特別檢查孔洞是否仍位於實體金屬上。",
    },
    balanced: {
      strength: 0.45,
      adapter: 0.72,
      steps: 30,
      hint: "平衡模式允許中度結構變化，適合先小量生成並人工比較。",
    },
    creative: {
      strength: 0.68,
      adapter: 0.58,
      steps: 36,
      hint: "高變化模式可能產生全新外觀，也更容易出現不可能的孔洞、螺絲與金屬融合，只適合探索。",
    },
  };
  const preset = presets[$("qualityMode").value];
  const flux = $("provider").value === "flux2_klein";
  $("strength").value = preset.strength;
  $("adapterScale").value = preset.adapter;
  $("steps").value = flux ? 4 : preset.steps;
  $("strength").disabled = flux;
  $("adapterScale").disabled = flux;
  $("steps").readOnly = flux;
  $("strengthValue").textContent = flux ? "由 FLUX 編輯指令控制" : preset.strength.toFixed(2);
  $("adapterValue").textContent = flux ? "原生參考圖編輯" : preset.adapter.toFixed(2);
  $("qualityHint").textContent = flux
    ? `FLUX.2 會直接理解參考圖與改造指令，固定使用 4 步推論。僅限非商業研究，所有輸出必須人工審查。${preset.hint}`
    : preset.hint;
}

function applyScenePreset() {
  const hints = {
    factory_mixed: "混合模式會依 seed 輪替五種工廠背景，方便建立分布較廣的訓練資料。",
    assembly_line: "金屬產線、治具、安全標線與頂部日光燈。",
    machine_enclosure: "CNC 機台金屬內壁、冷色工作燈與輕微油污。",
    maintenance_bench: "維修桌、少量工具、油漬及冷暖混合光。",
    conveyor_fixture: "輸送帶、定位夾具、導軌與方向性工業光。",
    warehouse_inspection: "倉儲品管站、側向自然光與工業頂燈。",
    custom: "不加入預設場景，只採用下方的自訂描述。",
  };
  $("sceneHint").textContent = hints[$("scenePreset").value];
}

function applyWorkpieceRecipe() {
  const recipe = WORKPIECE_RECIPES[$("workpieceRecipe").value];
  if (!recipe) return;
  $("scenePreset").value = "custom";
  $("prompt").value = `${OVERHEAD_FACTORY_BASE}\n\n${recipe}\n\n${FASTENER_REALISM}\n\n${FACTORY_REALISM}\n\n${OVERHEAD_COMPOSITION}`;
  $("negativePrompt").value = GENERATION_NEGATIVE;
  applyScenePreset();
}

async function refreshProject() {
  state.project = await api(`/api/projects/${state.project.id}`);
  renderProject();
}

async function startGeneration() {
  if (!$("generationReference").value) return toast("請先加入參考工件圖片。", true);
  if (["diffusers", "sdxl_controlnet", "flux2_klein"].includes($("provider").value) && !state.system.ml_ready) {
    return toast("真實 AI 尚未安裝，請先執行 setup.ps1 -WithML。", true);
  }
  if ($("provider").value === "sdxl_controlnet" && (state.system.vram_gb || 0) < 20) {
    return toast("SDXL + ControlNet 需要約 20 GB 以上 VRAM，請透過 SSH tunnel 使用學校 RTX A5000。", true);
  }
  if ($("provider").value === "flux2_klein" && (state.system.vram_gb || 0) < 20) {
    return toast("FLUX.2 Klein 需要學校 20 GB 以上 GPU 與 CPU offload，無法在本機 RTX 4060 執行。", true);
  }
  const seedValue = $("seed").value.trim();
  const payload = {
    reference_id: $("generationReference").value,
    prompt: `${$("prompt").value}\nSparse-hole constraint: use approximately 4 to 8 small empty holes only, with broad empty cast surfaces; the large central bearing opening is one flange feature and is not counted. Never create dense rows or grids of holes. Every screw must be fully seated in a matching hole or counterbore on the workpiece, with the head facing upward and the shaft hidden inside the hole; never place screws beside holes or standing vertically.`,
    negative_prompt: `${$("negativePrompt").value}, dense rows of holes, dense hole grid, more than eight small holes on the top face, repeated hole pattern, upside-down screw, inverted screw, vertical standing screw, loose screw beside a hole, screws on the conveyor, screws on the fixture, loose bolts in the background, fasteners outside the workpiece`,
    count: Number($("generationCount").value),
    seed: seedValue ? Number(seedValue) : null,
    strength: Number($("strength").value),
    ip_adapter_scale: Number($("adapterScale").value),
    guidance_scale: 6.5,
    steps: Number($("steps").value),
    quality_mode: $("qualityMode").value,
    scene_preset: $("scenePreset").value,
    framing: $("framing").value,
    provider: $("provider").value,
    prelabel: $("prelabel").checked,
    confidence: Number($("confidence").value),
    yolo_model_id: $("generationModel").value,
  };
  try {
    const job = await api(`/api/projects/${state.project.id}/generate`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    $("jobBox").hidden = false;
    $("startGeneration").disabled = true;
    pollJob(job.id);
  } catch (error) { toast(error.message, true); }
}

async function pollJob(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    const percent = job.total ? (job.completed / job.total) * 100 : 0;
    $("jobProgress").style.width = `${percent}%`;
    $("jobText").textContent = `${job.status} · ${job.completed} / ${job.total}`;
    if (job.status === "completed") {
      $("startGeneration").disabled = false;
      toast(`完成 ${job.completed} 張候選圖片。`);
      await loadCandidates();
      await refreshProject();
      return;
    }
    if (job.status === "failed") {
      $("startGeneration").disabled = false;
      return toast(job.error || "生成失敗。", true);
    }
    setTimeout(() => pollJob(jobId), 1200);
  } catch (error) {
    $("startGeneration").disabled = false;
    toast(error.message, true);
  }
}

async function loadCandidates() {
  if (!state.project) return;
  state.candidates = await api(`/api/projects/${state.project.id}/candidates`);
  filterCandidates();
  state.project.candidate_counts = {
    pending: state.candidates.filter((item) => item.status === "pending").length,
    approved: state.candidates.filter((item) => item.status === "approved").length,
    rejected: state.candidates.filter((item) => item.status === "rejected").length,
  };
  renderCounts();
}

function filterCandidates() {
  const filter = $("candidateFilter").value;
  state.filtered = filter === "all"
    ? state.candidates
    : state.candidates.filter((item) => item.status === filter);
  state.candidateIndex = Math.min(state.candidateIndex, Math.max(0, state.filtered.length - 1));
  loadCurrentCandidate();
}

function loadCurrentCandidate() {
  const candidate = state.filtered[state.candidateIndex];
  $("candidatePosition").textContent = candidate
    ? `${state.candidateIndex + 1} / ${state.filtered.length}`
    : `0 / ${state.filtered.length}`;
  $("canvasEmpty").hidden = Boolean(candidate);
  $("annotationCanvas").hidden = !candidate;
  if (!candidate) {
    state.boxes = [];
    state.selectedBoxId = null;
    renderSelectedBox();
    return;
  }
  state.boxes = structuredClone(candidate.boxes || []);
  state.selectedBoxId = null;
  state.image.onload = () => {
    const canvas = $("annotationCanvas");
    canvas.width = candidate.width;
    canvas.height = candidate.height;
    drawCanvas();
  };
  state.image.src = `/api/projects/${state.project.id}/candidates/${candidate.id}/image?t=${Date.now()}`;
  const generation = candidate.generation || {};
  $("generationInfo").innerHTML = `
    <strong>${escapeHtml(candidate.status)}</strong><br>
    Provider: ${escapeHtml(generation.provider || "-")}<br>
    Seed: ${escapeHtml(String(generation.seed ?? "-"))}<br>
    Strength: ${escapeHtml(String(generation.strength ?? "-"))}<br>
    IP scale: ${escapeHtml(String(generation.ip_adapter_scale ?? "-"))}<br>
    ${escapeHtml(generation.prompt || "")}`;
  renderSelectedBox();
}

function navigateCandidate(delta) {
  if (!state.filtered.length) return;
  state.candidateIndex = (state.candidateIndex + delta + state.filtered.length) % state.filtered.length;
  loadCurrentCandidate();
}

function canvasPoint(event) {
  const canvas = $("annotationCanvas");
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * canvas.width / rect.width,
    y: (event.clientY - rect.top) * canvas.height / rect.height,
    screenScale: canvas.width / rect.width,
  };
}

function boxAt(point) {
  return [...state.boxes].reverse().find((box) =>
    point.x >= box.x1 && point.x <= box.x2 && point.y >= box.y1 && point.y <= box.y2
  );
}

function cornerAt(box, point) {
  const tolerance = 11 * point.screenScale;
  const corners = {
    nw: [box.x1, box.y1], ne: [box.x2, box.y1],
    sw: [box.x1, box.y2], se: [box.x2, box.y2],
  };
  return Object.entries(corners).find(([, [x, y]]) =>
    Math.hypot(point.x - x, point.y - y) <= tolerance
  )?.[0] || null;
}

function canvasDown(event) {
  if (!state.filtered.length) return;
  const point = canvasPoint(event);
  const hit = boxAt(point);
  if (hit) {
    state.selectedBoxId = hit.id;
    state.drag = {
      mode: cornerAt(hit, point) ? "resize" : "move",
      corner: cornerAt(hit, point),
      start: point,
      original: structuredClone(hit),
    };
  } else {
    state.selectedBoxId = null;
    state.drag = {mode: "create", start: point, current: point};
  }
  renderSelectedBox();
  drawCanvas();
}

function canvasMove(event) {
  if (!state.drag) return;
  const point = canvasPoint(event);
  if (state.drag.mode === "create") {
    state.drag.current = point;
  } else {
    const box = state.boxes.find((item) => item.id === state.selectedBoxId);
    if (!box) return;
    const dx = point.x - state.drag.start.x;
    const dy = point.y - state.drag.start.y;
    const original = state.drag.original;
    if (state.drag.mode === "move") {
      const width = original.x2 - original.x1;
      const height = original.y2 - original.y1;
      box.x1 = clamp(original.x1 + dx, 0, state.image.width - width);
      box.y1 = clamp(original.y1 + dy, 0, state.image.height - height);
      box.x2 = box.x1 + width;
      box.y2 = box.y1 + height;
    } else {
      if (state.drag.corner.includes("n")) box.y1 = clamp(original.y1 + dy, 0, original.y2 - 2);
      if (state.drag.corner.includes("s")) box.y2 = clamp(original.y2 + dy, original.y1 + 2, state.image.height);
      if (state.drag.corner.includes("w")) box.x1 = clamp(original.x1 + dx, 0, original.x2 - 2);
      if (state.drag.corner.includes("e")) box.x2 = clamp(original.x2 + dx, original.x1 + 2, state.image.width);
    }
  }
  renderSelectedBox();
  drawCanvas();
}

function canvasUp(event) {
  if (!state.drag) return;
  if (state.drag.mode === "create") {
    const end = state.drag.current || canvasPoint(event);
    const x1 = Math.min(state.drag.start.x, end.x);
    const y1 = Math.min(state.drag.start.y, end.y);
    const x2 = Math.max(state.drag.start.x, end.x);
    const y2 = Math.max(state.drag.start.y, end.y);
    if (x2 - x1 > 5 && y2 - y1 > 5) {
      const box = {
        id: crypto.randomUUID().replaceAll("-", ""),
        class_id: Number($("annotationClass").value),
        x1, y1, x2, y2,
        confidence: null,
        source: "manual",
      };
      state.boxes.push(box);
      state.selectedBoxId = box.id;
    }
  }
  state.drag = null;
  renderSelectedBox();
  drawCanvas();
}

function drawCanvas() {
  const canvas = $("annotationCanvas");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (state.image.complete) context.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  for (const box of state.boxes) drawBox(context, box, box.id === state.selectedBoxId);
  if (state.drag?.mode === "create") {
    drawBox(context, {
      class_id: Number($("annotationClass").value),
      x1: Math.min(state.drag.start.x, state.drag.current.x),
      y1: Math.min(state.drag.start.y, state.drag.current.y),
      x2: Math.max(state.drag.start.x, state.drag.current.x),
      y2: Math.max(state.drag.start.y, state.drag.current.y),
    }, true);
  }
}

function drawBox(context, box, selected) {
  const color = COLORS[box.class_id % COLORS.length];
  context.save();
  context.strokeStyle = color;
  context.lineWidth = selected ? 4 : 3;
  context.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
  const label = state.project.classes[box.class_id] || `class ${box.class_id}`;
  const confidence = box.confidence == null ? "" : ` ${(box.confidence * 100).toFixed(0)}%`;
  context.font = "bold 17px sans-serif";
  const width = context.measureText(label + confidence).width + 14;
  context.fillStyle = color;
  context.fillRect(box.x1, Math.max(0, box.y1 - 26), width, 26);
  context.fillStyle = "#07110f";
  context.fillText(label + confidence, box.x1 + 7, Math.max(19, box.y1 - 7));
  if (selected) {
    for (const [x, y] of [[box.x1,box.y1],[box.x2,box.y1],[box.x1,box.y2],[box.x2,box.y2]]) {
      context.fillStyle = "#fff";
      context.fillRect(x - 5, y - 5, 10, 10);
      context.strokeRect(x - 5, y - 5, 10, 10);
    }
  }
  context.restore();
}

function renderSelectedBox() {
  const box = state.boxes.find((item) => item.id === state.selectedBoxId);
  $("deleteBox").disabled = !box;
  if (!box) {
    $("selectedBoxInfo").textContent = "未選取框";
    return;
  }
  $("annotationClass").value = String(box.class_id);
  $("selectedBoxInfo").textContent =
    `${state.project.classes[box.class_id]} · ${Math.round(box.x2-box.x1)}×${Math.round(box.y2-box.y1)} px` +
    (box.confidence == null ? "" : ` · ${(box.confidence * 100).toFixed(1)}%`);
}

function changeSelectedClass() {
  const box = state.boxes.find((item) => item.id === state.selectedBoxId);
  if (box) {
    box.class_id = Number($("annotationClass").value);
    box.source = "manual";
    box.confidence = null;
    renderSelectedBox();
    drawCanvas();
  }
}

function deleteSelectedBox() {
  if (!state.selectedBoxId) return;
  state.boxes = state.boxes.filter((box) => box.id !== state.selectedBoxId);
  state.selectedBoxId = null;
  renderSelectedBox();
  drawCanvas();
}

async function saveCandidate(status) {
  const candidate = state.filtered[state.candidateIndex];
  if (!candidate) return;
  if (status === "rejected") {
    try {
      await api(`/api/projects/${state.project.id}/candidates/${candidate.id}`, {method: "DELETE"});
      toast("圖片已刪除。");
      await loadCandidates();
    } catch (error) { toast(error.message, true); }
    return;
  }
  if (status === "approved" && candidate.generation?.training_eligible === false) {
    return toast("流程測試圖不能接受為訓練資料，請用真實 AI 模式生成。", true);
  }
  const qualityChecks = {};
  try {
    await api(`/api/projects/${state.project.id}/candidates/${candidate.id}`, {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({status, boxes: state.boxes, quality_checks: qualityChecks}),
    });
    toast(status === "approved" ? "已接受並儲存。" : status === "rejected" ? "已淘汰。" : "草稿已儲存。");
    await loadCandidates();
  } catch (error) { toast(error.message, true); }
}

function renderCounts() {
  const counts = state.project?.candidate_counts || {pending: 0, approved: 0, rejected: 0};
  $("candidateCounts").innerHTML = [
    ["待審核", counts.pending || 0],
    ["已接受", counts.approved || 0],
    ["已淘汰", counts.rejected || 0],
  ].map(([label, count]) => `<div class="count-card"><strong>${count}</strong>${label}</div>`).join("");
}

async function exportDataset() {
  try {
    const response = await api(`/api/projects/${state.project.id}/export`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({split_name: "train"}),
    });
    const blob = await response.blob();
    const disposition = response.headers.get("content-disposition") || "";
    const match = disposition.match(/filename="?([^"]+)"?/);
    const filename = match?.[1] || "workpiece-generated.yolov8.zip";
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
    toast("YOLOv8 ZIP 已匯出。");
  } catch (error) { toast(error.message, true); }
}

function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${name}`));
  if (name === "review") loadCandidates();
  if (name === "export") refreshProject();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  })[character]);
}
function clamp(value, minimum, maximum) { return Math.max(minimum, Math.min(maximum, value)); }

initialize();
