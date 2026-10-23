(() => {
  const SERVICE = "https://guarded-distillation-triage.onrender.com";
  const $ = (s) => document.querySelector(s);
  const pct = (x, d = 0) => (x == null ? "n/a" : (x * 100).toFixed(d) + "%");

  /* tabs */
  const tabs = [["#t-replay", "#p-replay"], ["#t-clinc", "#p-clinc"], ["#t-live", "#p-live"]];
  let timer = null;
  function select(i) {
    tabs.forEach(([t, p], k) => {
      $(t).setAttribute("aria-selected", k === i);
      $(t).tabIndex = k === i ? 0 : -1;
      $(p).hidden = k !== i;
    });
    history.replaceState(null, "", "#" + ["replay", "clinc", "live"][i]);
    const f = [$("#f-replay"), $("#f-clinc")][i];
    if (f && !f.src) f.src = f.dataset.src;
    clearInterval(timer);
    if (i === 2) { poll(); timer = setInterval(poll, 10000); }
  }
  tabs.forEach(([t], i) => {
    $(t).addEventListener("click", () => select(i));
    $(t).addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const n = (i + 1) % tabs.length; select(n); $(tabs[n][0]).focus(); }
    });
  });
  [$("#f-replay"), $("#f-clinc")].forEach((frame) => {
    frame.addEventListener("load", () => {
      const fit = () => { try { frame.style.height = frame.contentDocument.documentElement.scrollHeight + 8 + "px"; } catch (e) {} };
      fit(); setInterval(fit, 1500);
    });
  });

  /* live view */
  const card = (v, label, cls = "", extra = "") => `<div class="card ${cls}"><b>${v}</b><span>${label}</span>${extra}</div>`;
  async function poll() {
    const banner = $("#live-banner");
    try {
      const [m, a] = await Promise.all([fetch(SERVICE + "/metrics").then((r) => r.json()), fetch(SERVICE + "/alerts").then((r) => r.json())]);
      const over = m.leak_over_budget, leak = m.estimated_leak_rate, bud = m.leak_budget;
      const alerts = a.alerts || [];
      banner.className = "banner " + (alerts.length ? "bad" : "ok");
      banner.textContent = alerts.length ? "Needs a look: " + alerts.map((x) => x.detail).join("; ") : "Healthy. No stress spike, leaks within budget, spend under the cap.";
      $("#live-grid").innerHTML =
        card(m.model_version, "model version in use") +
        card(m.decisions_total.toLocaleString(), "questions routed so far") +
        card(pct(m.escalation_rate), `sent to Claude, last ${m.window} questions`) +
        card(pct(leak, 1), `estimated wrong answers that slipped through, against a budget of ${pct(bud, 1)}`, over ? "bad" : "",
             `<div class="meter-bar"><i class="${over ? "over" : ""}" style="width:${Math.min(100, (leak / bud) * 100)}%"></i></div>`) +
        card(pct(m.audit.rate, 1), `of ${m.audit.audited} audited answers disagreed with Claude`) +
        card("$" + m.spend.today_usd.toFixed(3), `spent today of a $${m.spend.cap_usd.toFixed(2)} cap (${m.spend.cache_hits} cached, ${m.spend.api_calls} live calls)`) +
        card(m.promotions + " / " + m.rollbacks, "model promotions / automatic rollbacks");
      const top = m.stress.top || [];
      $("#live-stress").innerHTML = top.length
        ? top.map((t) => `<div class="stress-row"><span>${t.intent}</span><div class="tr"><i class="${t.stress >= m.stress.spike_threshold ? "hot" : ""}" style="width:${Math.min(100, t.stress * 100)}%"></i></div><span>${t.stress.toFixed(2)}</span></div>`).join("")
        : '<p class="small">No intent is under stress.</p>';
      $("#live-foot").textContent = `Uptime ${Math.round(m.uptime_s / 60)} min. The server's log and spend ledger reset when the free instance restarts.`;
      $("#live-ts").textContent = "Updated " + new Date().toLocaleTimeString();
    } catch (e) {
      banner.className = "banner warn";
      banner.textContent = "Waiting for the server. A free instance sleeps when idle; the first request can take up to a minute. Retrying every 10 seconds.";
    }
  }
  $("#try-go").addEventListener("click", async () => {
    const out = $("#try-out");
    out.textContent = "working...";
    try {
      const r = await fetch(SERVICE + "/triage", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text: $("#try-q").value }) });
      const j = await r.json();
      out.textContent = j.intent ? `${j.intent}  (${j.decision === "accept" ? "answered by the small model" : "sent to Claude"}, team: ${j.team}, urgency: ${j.urgency})` : JSON.stringify(j);
    } catch (e) { out.textContent = "The server did not answer. It may be waking up; try again in a minute."; }
    poll();
  });

  select({ "#clinc": 1, "#live": 2 }[location.hash] || 0);
})();
