const DATA_URL = "data/public_dashboard.json";

const $ = (id) => document.getElementById(id);

function text(value, fallback = "--") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function escapeHtml(value) {
  return text(value, "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function numeric(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function integer(value) {
  const parsed = numeric(value);
  return parsed === null ? "--" : String(Math.round(parsed));
}

function percent(value) {
  const parsed = numeric(value);
  if (parsed === null) return "--";
  return `${parsed.toFixed(1)}%`;
}

function ratioPercent(value) {
  const parsed = numeric(value);
  if (parsed === null) return "--";
  return `${(parsed * 100).toFixed(1)}%`;
}

function signedPercent(value) {
  const parsed = numeric(value);
  if (parsed === null) return "--";
  const sign = parsed > 0 ? "+" : "";
  return `${sign}${parsed.toFixed(1)}%`;
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

function isPaperSignal(game) {
  const raw = String(game.recommendation_raw || "").toUpperCase();
  return raw && raw !== "NO BET" && raw !== "TRACKING ONLY";
}

function renderNav(items = []) {
  const fallback = [
    { label: "Today", href: "#games" },
    { label: "Record", href: "#record" },
    { label: "Governance", href: "#governance" },
    { label: "Roadmap", href: "#roadmap" },
  ];
  const links = (items.length ? items : fallback)
    .map((item) => `<a href="${escapeHtml(item.href)}">${escapeHtml(item.label)}</a>`)
    .join("");
  $("navLinks").innerHTML = links;
}

function renderHero(data) {
  $("heroTitle").textContent = text(data.hero?.title, "MLB Paper Prediction Board");
  $("heroSubtitle").textContent = text(data.hero?.subtitle, "Paper-only MLB research board with model governance and settlement tracking.");
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
  const games = data.games || [];
  const paperSignals = games.filter(isPaperSignal).length;
  const noBets = Math.max(0, games.length - paperSignals);
  const rolling30 = data.performance?.rolling_windows?.["30d"]?.accuracy;
  $("gameCount").textContent = text(data.metrics?.game_count ?? data.metrics?.scheduled_game_count);
  $("paperSignalCount").textContent = integer(paperSignals);
  $("noBetCount").textContent = integer(noBets);
  $("sampleCount").textContent = text(data.performance?.official_accuracy?.sample_count ?? data.metrics?.clean_settled_sample_count);
  $("rolling30Accuracy").textContent = ratioPercent(rolling30);
  $("positiveClv").textContent = percent(data.metrics?.positive_clv_rate_pct);
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
      const flags = (game.risk_flags || []).slice(0, 5).map((flag) => `<span class="tag">${escapeHtml(flag)}</span>`).join("");
      const note = (game.public_notes || [])[0] || "Tracking only. No betting recommendation.";
      const side = sideContext(game);
      const projectedLabel = side.projectedSide
        ? `${escapeHtml(side.projectedSide.team)} ${percent(side.projectedSide.prob)}`
        : "--";
      const valueLabel = side.valueLean
        ? `${escapeHtml(side.valueLean.team)} ${signedPercent(side.valueLean.edge)}`
        : "--";
      const homeAwayLine = `Home ${percent(side.homeProb)} · Away ${percent(side.awayProb)}`;
      const marketLine = `Market home ${percent(game.market_home_probability_pct)} · Model edge home ${signedPercent(side.homeEdge)}`;
      const paperTag = isPaperSignal(game) ? escapeHtml(game.recommendation_raw) : "No Bet";
      const pitchers = `${escapeHtml(game.away_probable_pitcher_name || "Away pitcher TBD")} vs ${escapeHtml(game.home_probable_pitcher_name || "Home pitcher TBD")}`;

      return `
        <article class="game-card">
          <div class="game-mainline">
            <div>
              <div class="game-top">
                <span class="status-pill ${statusClass}">${escapeHtml(game.recommendation_label || "TRACKING ONLY")}</span>
                <span>${formatTime(game.start_time)}</span>
              </div>
              <h3>${escapeHtml(game.away_team)} <span>@</span> ${escapeHtml(game.home_team)}</h3>
              <p class="match-meta">${pitchers}<br>${escapeHtml(game.moneyline_gate_status || "tracking gate")}</p>
            </div>
            <div>
              <div class="prob-row">
                <div class="prob-box"><strong>${projectedLabel}</strong><span>Projected side</span></div>
                <div class="prob-box"><strong>${valueLabel}</strong><span>Value lean vs market</span></div>
              </div>
              <p class="game-note">${escapeHtml(note)}<br><small>${homeAwayLine}</small><br><small>${marketLine}</small></p>
            </div>
          </div>
          <div class="tag-row">
            <span class="tag strong">${paperTag}</span>
            <span class="tag">Grade ${escapeHtml(game.data_quality_grade || "--")}</span>
            <span class="tag">${escapeHtml(game.lineup_status || "lineup unknown")}</span>
            <span class="tag">${escapeHtml(game.pitcher_status || "pitcher unknown")}</span>
            ${flags}
          </div>
        </article>
      `;
    })
    .join("");
}

function renderRecord(data) {
  const official = data.performance?.official_accuracy || {};
  const rolling = data.performance?.rolling_windows || {};
  const slices = data.performance?.slices || {};
  const pending = data.performance?.pending_predictions || {};
  const items = [
    ["✓ / ✗", `${integer(official.correct)} / ${integer((numeric(official.sample_count) || 0) - (numeric(official.correct) || 0))}`, "official settled record", "record-wide"],
    ["Official accuracy", ratioPercent(official.accuracy), `${integer(official.sample_count)} settled games`, ""],
    ["7D accuracy", ratioPercent(rolling["7d"]?.accuracy), `${integer(rolling["7d"]?.sample_count)} samples`, ""],
    ["30D accuracy", ratioPercent(rolling["30d"]?.accuracy), `${integer(rolling["30d"]?.sample_count)} samples`, ""],
    ["Pending", integer(pending.count), escapeHtml(pending.reason || "unsettled"), ""],
    ["Paper signals", ratioPercent(slices.paper_signals?.accuracy), `${integer(slices.paper_signals?.sample_count)} samples`, ""],
  ];
  $("recordBoard").innerHTML = items
    .map(([label, value, caption, extra]) => `
      <article class="record-item ${extra}">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
        <small>${caption}</small>
      </article>
    `)
    .join("");
}

function renderPerformance(data) {
  const official = data.performance?.official_accuracy || {};
  const clv = data.performance?.clv_metrics || {};
  const metrics = data.metrics || {};
  const items = [
    ["Brier", text(official.brier ?? metrics.model_brier), "lower is better"],
    ["Logloss", text(official.logloss), "probability quality"],
    ["Model AUC", text(metrics.model_auc), "current eval"],
    ["CLV avg", text(clv.avg_clv ?? metrics.avg_clv), "price movement only"],
    ["CLV samples", integer(clv.sample_count), "tracked odds"],
    ["Data errors", integer(metrics.data_contract_errors), "contract gate"],
  ];
  $("performanceGrid").innerHTML = items
    .map(([label, value, caption]) => `
      <article class="performance-item">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
        <small>${escapeHtml(caption)}</small>
      </article>
    `)
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
          <div><strong>${escapeHtml(label)}: ${escapeHtml(text(status))}</strong><small>${escapeHtml(caption)}</small></div>
        </div>
      `;
    })
    .join("");

  const artifact = data.governance_summary?.artifact_quarantine || {};
  $("artifactCard").innerHTML = `
    <dl>
      <div><dt>Status</dt><dd>${escapeHtml(text(artifact.status))}</dd></div>
      <div><dt>Artifact samples</dt><dd>${escapeHtml(text(artifact.artifact_training_sample_count))}</dd></div>
      <div><dt>Current samples</dt><dd>${escapeHtml(text(artifact.current_training_sample_count))}</dd></div>
      <div><dt>Production minimum</dt><dd>${escapeHtml(text(artifact.minimum_production_training_samples))}</dd></div>
      <div><dt>Stale mismatch</dt><dd>${escapeHtml(text(artifact.stale_sample_mismatch))}</dd></div>
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
      const features = (action.top_features || []).map((feature) => `<li>${escapeHtml(feature)}</li>`).join("");
      return `
        <article class="roadmap-card">
          <strong>${escapeHtml(action.feature_group || "Feature group")}</strong>
          <p>${escapeHtml(action.rationale || "Review and backfill this feature group.")}</p>
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
    renderRecord(data);
    renderPerformance(data);
    renderGovernance(data);
    renderRoadmap(data);
  } catch (error) {
    $("gameGrid").innerHTML = `<div class="error-state">Public dashboard data unavailable: ${escapeHtml(error.message)}</div>`;
    $("heroTitle").textContent = "Dashboard data unavailable";
    $("heroSubtitle").textContent = "Run the data builder or wait for the next scheduled update.";
  }
}

loadDashboard();
