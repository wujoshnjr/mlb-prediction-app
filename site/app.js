const DATA_URL = "data/public_dashboard.json";

const $ = (id) => document.getElementById(id);

function text(value, fallback = "--") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function numeric(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function percent(value) {
  const parsed = numeric(value);
  if (parsed === null) return "--";
  return `${parsed.toFixed(1)}%`;
}

function signedPercent(value) {
  const parsed = numeric(value);
  if (parsed === null) return "--";
  const sign = parsed > 0 ? "+" : "";
  return `${sign}${parsed.toFixed(1)}%`;
}

function sideContext(game) {
  const homeProb = numeric(game.home_win_probability_pct);
  const awayProb = homeProb === null ? null : 100 - homeProb;
  const homeEdge = numeric(game.model_edge_home_pct);
  const awayEdge = homeEdge === null ? null : -homeEdge;

  const projectedSide = homeProb === null
    ? null
    : homeProb >= 50
      ? { team: game.home_team, side: "Home", prob: homeProb }
      : { team: game.away_team, side: "Away", prob: awayProb };

  const valueLean = homeEdge === null
    ? null
    : homeEdge >= 0
      ? { team: game.home_team, side: "Home", edge: homeEdge }
      : { team: game.away_team, side: "Away", edge: awayEdge };

  return { homeProb, awayProb, homeEdge, awayEdge, projectedSide, valueLean };
}

function classForStatus(value) {
  const status = String(value || "").toLowerCase();
  if (["failed", "error", "fatal", "danger"].some((word) => status.includes(word))) return "danger";
  if (["blocked", "warning", "quarantined", "insufficient", "tracking"].some((word) => status.includes(word))) return "warning";
  if (["ok", "completed", "ready"].some((word) => status.includes(word))) return "ok";
  return "neutral";
}

function formatTime(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return text(value);
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderNav(items = []) {
  const links = items.map((item) => `<a href="${item.href}">${item.label}</a>`).join("");
  $("navLinks").innerHTML = links || "";
}

function renderHero(data) {
  $("heroTitle").textContent = text(data.hero?.title, "MLB Intelligence Cloud");
  $("heroSubtitle").textContent = text(data.hero?.subtitle, "AI research board for MLB games.");
  $("publicDisclaimer").textContent = text(data.public_disclaimer);

  const status = data.hero?.primary_status || data.system_status?.model_quality_status || "tracking only";
  const badge = data.hero?.status_badge || data.system_status?.badge || classForStatus(status);
  const primaryStatus = $("primaryStatus");
  primaryStatus.textContent = status;
  primaryStatus.className = `status-pill ${badge}`;

  $("pipelineStatus").textContent = text(data.system_status?.pipeline_health_status);
  $("modelStatus").textContent = text(data.system_status?.model_quality_status);
  $("contractStatus").textContent = text(data.system_status?.data_contract_status);
  $("updatedAt").textContent = formatTime(data.source_generated_at || data.generated_at);
}

function renderMetrics(data) {
  $("gameCount").textContent = text(data.metrics?.game_count ?? data.metrics?.scheduled_game_count);
  $("sampleCount").textContent = text(data.metrics?.clean_settled_sample_count);
  $("positiveClv").textContent = percent(data.metrics?.positive_clv_rate_pct);
  $("repoErrors").textContent = text(data.metrics?.repo_anomaly_errors, "0");
}

function renderGames(games = []) {
  const container = $("gameGrid");
  if (!games.length) {
    container.innerHTML = `<div class="empty-state">目前沒有可顯示的賽事。資料更新後會自動出現在這裡。</div>`;
    return;
  }

  container.innerHTML = games
    .map((game) => {
      const statusClass = classForStatus(`${game.recommendation_status} ${game.risk_profile}`);
      const flags = (game.risk_flags || []).slice(0, 4).map((flag) => `<span class="tag">${flag}</span>`).join("");
      const note = (game.public_notes || [])[0] || "Tracking only. No betting recommendation.";
      const side = sideContext(game);
      const projectedLabel = side.projectedSide
        ? `${text(side.projectedSide.team)} ${percent(side.projectedSide.prob)}`
        : "--";
      const valueLabel = side.valueLean
        ? `${text(side.valueLean.team)} ${signedPercent(side.valueLean.edge)}`
        : "--";
      const homeAwayLine = `Home ${percent(side.homeProb)} · Away ${percent(side.awayProb)}`;
      const valueLine = `Value lean is not a bet. Home edge ${signedPercent(side.homeEdge)} · Away edge ${signedPercent(side.awayEdge)}.`;

      return `
        <article class="game-card">
          <div class="game-top">
            <span class="status-pill ${statusClass}">${text(game.recommendation_label, "TRACKING ONLY")}</span>
            <span>${formatTime(game.start_time)}</span>
          </div>
          <h3>${text(game.away_team)} <span>@</span> ${text(game.home_team)}</h3>
          <div class="prob-row">
            <div class="prob-box"><strong>${projectedLabel}</strong><span>Projected side</span></div>
            <div class="prob-box"><strong>${valueLabel}</strong><span>Value lean vs market</span></div>
          </div>
          <p class="game-note">${note}<br><small>${homeAwayLine}</small><br><small>${valueLine}</small></p>
          <div class="tag-row">
            <span class="tag">No bet</span>
            <span class="tag">Grade ${text(game.data_quality_grade)}</span>
            <span class="tag">${text(game.lineup_status, "lineup unknown")}</span>
            <span class="tag">${text(game.pitcher_status, "pitcher unknown")}</span>
            ${flags}
          </div>
        </article>
      `;
    })
    .join("");
}

function renderGovernance(data) {
  const system = data.system_status || {};
  const items = [
    ["Pipeline health", system.pipeline_health_status, "Data and report pipeline is structurally readable."],
    ["Model quality", system.model_quality_status, "Promotion remains locked until baseline, walk-forward, and collapse gates improve."],
    ["Data contract", system.data_contract_status, "Public site consumes validated JSON outputs."],
    ["Repo anomaly", system.repo_anomaly_status, "Scanner checks syntax, JSON, suspicious strings, workflows, and safety flags."],
    ["Live betting", system.live_betting_allowed ? "enabled" : "disabled", "Must remain disabled."],
    ["Automated wagering", system.automated_wagering_allowed ? "enabled" : "disabled", "Must remain disabled."],
  ];

  $("governanceTimeline").innerHTML = items
    .map(([label, status, caption]) => {
      const cls = classForStatus(`${label} ${status}`);
      const normalized = label.includes("Live") || label.includes("Automated") ? "ok" : cls;
      return `
        <div class="timeline-item">
          <span class="timeline-dot ${normalized}"></span>
          <div><strong>${label}: ${text(status)}</strong><small>${caption}</small></div>
        </div>
      `;
    })
    .join("");

  const artifact = data.governance_summary?.artifact_quarantine || {};
  $("artifactCard").innerHTML = `
    <dl>
      <div><dt>Status</dt><dd>${text(artifact.status)}</dd></div>
      <div><dt>Artifact samples</dt><dd>${text(artifact.artifact_training_sample_count)}</dd></div>
      <div><dt>Current samples</dt><dd>${text(artifact.current_training_sample_count)}</dd></div>
      <div><dt>Production minimum</dt><dd>${text(artifact.minimum_production_training_samples)}</dd></div>
      <div><dt>Stale mismatch</dt><dd>${text(artifact.stale_sample_mismatch)}</dd></div>
    </dl>
  `;
}

function renderRoadmap(data) {
  const actions = data.feature_roadmap?.actions || [];
  const container = $("roadmapGrid");
  if (!actions.length) {
    container.innerHTML = `<div class="empty-state">Feature roadmap 尚未產出。下一次治理報告會自動補上。</div>`;
    return;
  }

  container.innerHTML = actions
    .map((action) => {
      const features = (action.top_features || []).map((feature) => `<li>${feature}</li>`).join("");
      return `
        <article class="roadmap-card">
          <strong>${text(action.feature_group, "Feature group")}</strong>
          <p>${text(action.rationale, "Review and backfill this feature group.")}</p>
          <ul>${features}</ul>
        </article>
      `;
    })
    .join("");
}

async function loadDashboard() {
  try {
    const response = await fetch(DATA_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`Dashboard data returned ${response.status}`);
    const data = await response.json();
    renderNav(data.navigation || []);
    renderHero(data);
    renderMetrics(data);
    renderGames(data.games || []);
    renderGovernance(data);
    renderRoadmap(data);
  } catch (error) {
    $("gameGrid").innerHTML = `<div class="error-state">Public dashboard data unavailable: ${error.message}</div>`;
    $("heroTitle").textContent = "Dashboard data unavailable";
    $("heroSubtitle").textContent = "Run the data builder or wait for the next scheduled update.";
  }
}

loadDashboard();
