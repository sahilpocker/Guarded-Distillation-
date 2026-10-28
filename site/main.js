(() => {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  /* ---------- savings calculator ---------- */
  const COST_RATIO = 0.512;             // measured: $0.021 / $0.041 per 1,000 questions, audits included
  function calc() {
    const vol = $("#c-vol"), price = $("#c-price");
    const usd = (x) => "$" + (x >= 100 ? Math.round(x).toLocaleString("en-US") : x.toFixed(2));
    function draw() {
      const all = Math.max(0, +vol.value || 0) / 1000 * Math.max(0, +price.value || 0);
      $("#c-all").textContent = usd(all);
      $("#c-sys").textContent = usd(all * COST_RATIO);
      $("#c-save").textContent = usd(all * (1 - COST_RATIO));
    }
    vol.addEventListener("input", draw);
    price.addEventListener("input", draw);
    draw();
  }

  /* ---------- story stage ---------- */
  const OLD = ["Where is my card?", "Why was I charged twice?", "How do I change my PIN?", "Exchange rate for euros?", "My top-up failed", "Transfer still pending", "Card won't work at the ATM", "How long do transfers take?", "I lost my phone", "Refund not showing up", "Fee to withdraw cash?", "Can't verify my identity"];
  const NEW = ["Can I get a virtual card?", "How do disposable cards work?", "My virtual card isn't working", "Does it work with Apple Pay?", "Disposable card limits?"];
  // Per step: cheap = share of ordinary questions the small model takes; newShare = share of traffic about the new product;
  // newCheap = chance the small model takes a new-product question; wrongNew / wrongOld = chance it is wrong when it answers.
  const STEPS = [
    { cheap: 0,   gate: 0, newShare: 0,    newCheap: 0,    wrongNew: 0,    wrongOld: 0,    learn: 0,    learnNew: 0,   note: "hasn't learned anything yet",
      cap: "Every question is sent to the big AI. Accurate, and the cost grows with every message." },
    { cheap: 0,   gate: 0, newShare: 0,    newCheap: 0,    wrongNew: 0,    wrongOld: 0,    learn: 0.62, learnNew: 0,   note: "learning from the big AI's answers", copy: true,
      cap: "Each answer is copied to the small model as a lesson." },
    { cheap: 0.5, gate: 1, newShare: 0,    newCheap: 0,    wrongNew: 0,    wrongOld: 0.02, learn: 1,    learnNew: 0,   note: "knows 77 topics",
      cap: "The gate lets the small model answer only when it is sure. The rest still go to the big AI." },
    { cheap: 0.5, gate: 1, newShare: 0.24, newCheap: 0.92, wrongNew: 0.62, wrongOld: 0.02, learn: 1,    learnNew: 0,   note: "has never seen virtual cards", alert: 1,
      cap: "New-product questions look like old ones, so the small model answers them confidently, and gets many wrong." },
    { cheap: 0.5, gate: 1, newShare: 0.24, newCheap: 0.1,  wrongNew: 0.12, wrongOld: 0.02, learn: 1,    learnNew: 0.9, note: "learning virtual cards", copy: true, careful: 1,
      cap: "The gate sends the new topic back to the big AI, and the small model learns from those answers." },
    { cheap: 0.5, gate: 1, newShare: 0.24, newCheap: 0.55, wrongNew: 0.08, wrongOld: 0.02, learn: 1,    learnNew: 1,   note: "knows 77 topics + virtual cards", careful: 1,
      cap: "New topics take a little longer to hand over. Mistakes stay low and so does the AI bill." },
  ];
  let cur = 0, running = false, spawnAcc = 0, activeChips = 0;
  const win = [];                       // rolling window of the last 40 outcomes
  const stage = $("#stage"), board = $("#board");
  const el = { gate: $("#gate"), ai: $("#node-ai"), sm: $("#node-small"), cap: $("#caption"), mAi: $("#m-ai"), mWrong: $("#m-wrong"),
    barOld: $("#bar-old"), barNew: $("#bar-new"), barNewWrap: $("#bar-new-wrap"), note: $("#small-note") };

  function setMeters() {
    if (!win.length) return;
    const n = win.length, ai = win.filter(w => w.ai).length, wr = win.filter(w => w.wrong).length;
    el.mAi.textContent = Math.round(ai / n * 100) + "%";
    el.mWrong.textContent = (wr / n * 100).toFixed(0);
  }
  function applyStep(i) {
    cur = i;
    const s = STEPS[i];
    $$(".step").forEach((d, k) => d.classList.toggle("on", k === i));
    el.cap.textContent = s.cap;
    el.note.textContent = s.note;
    el.gate.classList.toggle("on", !!s.gate);
    el.gate.classList.toggle("alert", !!s.alert);
    el.gate.textContent = s.alert ? "too trusting" : s.careful ? "cautious" : "gate";
    const dur = reduce ? 0 : 0.9;
    gsap.to(el.barOld, { width: s.learn * 100 + "%", duration: dur, ease: "power2.out" });
    gsap.to(el.barNewWrap, { opacity: s.newShare ? 1 : 0, duration: dur });
    gsap.to(el.barNew, { width: s.learnNew * 100 + "%", duration: dur, ease: "power2.out" });
    win.length = 0;
    // expected steady state, shown immediately (and as the only value when motion is reduced)
    const smallShare = (1 - s.newShare) * s.cheap + s.newShare * s.newCheap;
    el.mAi.textContent = Math.round((1 - smallShare) * 100) + "%";
    el.mWrong.textContent = Math.round(((1 - s.newShare) * s.cheap * s.wrongOld + s.newShare * s.newCheap * s.wrongNew) * 100);
  }

  function point(node, edge) {
    const b = board.getBoundingClientRect(), r = node.getBoundingClientRect();
    return { x: r.left - b.left + (edge === "left" ? 0 : r.width / 2), y: r.top - b.top + r.height / 2 };
  }
  function spawn() {
    if (activeChips > 16) return;
    const s = STEPS[cur];
    const isNew = Math.random() < s.newShare;
    const pool = isNew ? NEW : OLD;
    const text = pool[Math.floor(Math.random() * pool.length)];
    const toSmall = Math.random() < (isNew ? s.newCheap : s.cheap);
    const wrong = toSmall && Math.random() < (isNew ? s.wrongNew : s.wrongOld);
    const chip = document.createElement("div");
    chip.className = "chip" + (isNew ? " new" : "");
    chip.textContent = text;
    board.appendChild(chip);
    activeChips++;
    const g = point(el.gate, "c"), dest = point(toSmall ? el.sm : el.ai, "left");
    gsap.set(chip, { x: 34, y: board.clientHeight * (0.25 + Math.random() * 0.5) });
    const tl = gsap.timeline({ onComplete: () => { chip.remove(); activeChips--; } });
    tl.to(chip, { x: g.x - 40, y: g.y - 10 + (Math.random() - 0.5) * 24, duration: 1.1, ease: "none" });
    tl.add(() => {
      win.push({ ai: !toSmall, wrong });
      if (win.length > 40) win.shift();
      setMeters();
      if (toSmall) chip.classList.add(wrong ? "wrong" : "right");
      if (s.copy && !toSmall) {          // the big AI's answer is copied to the small model as a lesson
        const c = chip.cloneNode(true);
        board.appendChild(c);
        c.classList.add("right");
        const t = point(el.sm, "left");
        gsap.set(c, { x: dest.x - 60, y: dest.y });
        gsap.to(c, { x: t.x - 40, y: t.y, opacity: 0, scale: 0.6, duration: 1.2, ease: "power1.in", delay: 0.6, onComplete: () => c.remove() });
      }
    });
    tl.to(chip, { x: dest.x - 60, y: dest.y - 10 + (Math.random() - 0.5) * 30, duration: 0.9, ease: "power1.inOut" });
    tl.to(chip, { opacity: 0, duration: 0.5, delay: wrong ? 0.7 : 0.15 });
  }
  function loop() {
    if (!running || document.hidden) return;
    spawnAcc += gsap.ticker.deltaRatio() / 60;
    if (spawnAcc > 0.55) { spawnAcc = 0; spawn(); }
  }

  function story() {
    if (!window.gsap || !window.ScrollTrigger) return;
    gsap.registerPlugin(ScrollTrigger);
    applyStep(0);
    $$(".step").forEach((d, i) => {
      ScrollTrigger.create({ trigger: d, start: "top 62%", end: "bottom 38%", onToggle: self => { if (self.isActive) applyStep(i); } });
    });
    if (!reduce) {
      new IntersectionObserver(es => { running = es[0].isIntersecting; }, { threshold: 0.2 }).observe(stage);
      gsap.ticker.add(loop);
    }
  }

  /* ---------- result bars ---------- */
  function bars() {
    const fills = $$(".fill");
    if (!window.gsap || reduce) { fills.forEach(f => (f.style.width = f.dataset.w + "%")); return; }
    fills.forEach(f => ScrollTrigger.create({ trigger: f, start: "top 90%", once: true,
      onEnter: () => gsap.to(f, { width: f.dataset.w + "%", duration: 1.1, ease: "power3.out" }) }));
  }

  /* ---------- dial ---------- */
  // Static gate + retraining + class balancing, MiniLM student; mean of six replays (seeds 0-5); a = the gate's leak budget.
  const DIAL = [
    { a: 0.03,   calls: 52.2, acc: 0.60, errors: 39.8 }, { a: 0.025,  calls: 54.4, acc: 0.57, errors: 35.1 },
    { a: 0.02,   calls: 57.1, acc: 0.46, errors: 29.6 }, { a: 0.0175, calls: 58.7, acc: 0.42, errors: 27.0 },
    { a: 0.015,  calls: 60.7, acc: 0.40, errors: 23.4 }, { a: 0.0125, calls: 63.0, acc: 0.36, errors: 21.0 },
    { a: 0.01,   calls: 67.4, acc: 0.32, errors: 15.7 },
  ];
  function dial() {
    const wrap = $("#squares"), range = $("#dial-range");
    for (let i = 0; i < 100; i++) wrap.appendChild(document.createElement("i"));
    const cells = [...wrap.children];
    function draw() {
      const d = DIAL[+range.value];
      const ai = Math.round(d.calls), err = Math.max(1, Math.round(d.errors / 10));
      cells.forEach((c, i) => { c.className = i < ai ? "" : i >= 100 - err ? "err" : "cheap"; });
      $("#d-calls").textContent = Math.round(d.calls) + "%";
      $("#d-cost").textContent = "$" + (d.calls / 100 * 0.041).toFixed(3);
      $("#d-err").textContent = Math.round(d.errors);
      $("#d-acc").textContent = (d.acc >= 0 ? "+" : "") + d.acc.toFixed(1);
    }
    range.addEventListener("input", draw);
    draw();
  }

  /* ---------- routing chart (from routing-data.json) ---------- */
  const ROUTE_COLORS = { margin: "var(--ai)", maxprob: "var(--ink-3)", learned: "var(--cheap)", learned_sem: "var(--cheap-2)" };
  const ROUTE_LABEL = { margin: "Old rule (margin)", maxprob: "Max probability", learned: "Learned rule", learned_sem: "Learned rule + topic stress" };
  const FAM_NAME = { banking: "Banking launch", credit_cards: "New domain: credit cards", travel: "New domain: travel" };
  async function routing() {
    const box = $("#route-box");
    let data;
    try { data = await fetch("routing-data.json").then((r) => r.json()); } catch (e) { box.textContent = "Chart data did not load."; return; }
    let fam = "banking", metric = "acc", show = "two";
    const W = 760, H = 400, L = 62, T = 16, B = 50, R = 16;
    function draw() {
      const key = metric === "acc" ? "acc" : "l600";
      const ref = data.llm[fam][key];
      const rules = show === "two" ? ["margin", "learned"] : Object.keys(ROUTE_COLORS);
      const lines = rules.map((r) => [r, data.curves[fam + "|" + r].map((p) => ({ calls: p.calls, d: p[key] - ref })).sort((a, b) => a.calls - b.calls)]);
      const all = lines.flatMap(([, p]) => p.map((q) => q.d)).concat(0);
      const x0 = 20, x1 = 100, y0 = Math.floor(Math.min(...all) - 0.5), y1 = Math.ceil(Math.max(...all) + 0.5);
      const X = (v) => L + (v - x0) / (x1 - x0) * (W - L - R), Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
      const unit = metric === "acc" ? "correct answers" : "correct answers on new-product questions";
      let o = `<svg id="route-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Points of ${unit} compared with asking the big AI on every question, against share of questions sent to the big AI">`;
      for (let v = 20; v <= 100; v += 10) o += `<line class="grid" x1="${X(v)}" x2="${X(v)}" y1="${T}" y2="${H - B}"/><text x="${X(v)}" y="${H - B + 17}" text-anchor="middle">${v}%</text>`;
      const step = (y1 - y0) > 40 ? 10 : (y1 - y0) > 16 ? 5 : (y1 - y0) > 8 ? 2 : 1;
      for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) o += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${v > 0 ? "+" : ""}${v}</text>`;
      o += `<line class="zero" x1="${L}" x2="${W - R}" y1="${Y(0)}" y2="${Y(0)}"/><text class="zl" x="${W - R - 4}" y="${Y(0) - 6}" text-anchor="end">same as asking the big AI every time</text>`;
      o += `<text x="${(L + W - R) / 2}" y="${H - 8}" text-anchor="middle">Share of questions sent to the big AI (further left = cheaper)</text>`;
      o += `<text transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">Points of ${unit}, compared with the big AI</text>`;
      lines.forEach(([r, pts]) => {
        o += `<path d="M${pts.map((p) => `${X(p.calls).toFixed(1)} ${Y(p.d).toFixed(1)}`).join("L")}" fill="none" stroke="${ROUTE_COLORS[r]}" stroke-width="${r === "learned" ? 3.2 : 2.2}" stroke-linejoin="round"/>`;
        pts.forEach((p) => { o += `<circle cx="${X(p.calls)}" cy="${Y(p.d)}" r="3.4" fill="${ROUTE_COLORS[r]}"><title>${ROUTE_LABEL[r]}: ${p.calls.toFixed(0)}% of questions to the big AI, ${p.d >= 0 ? "+" : ""}${p.d.toFixed(2)} points</title></circle>`; });
      });
      o += "</svg>";
      box.innerHTML = o + `<div class="route-legend">${rules.map((r) => `<span><i style="background:${ROUTE_COLORS[r]}"></i>${ROUTE_LABEL[r]}</span>`).join("")}</div>`;
    }
    const seg = (name, opts, get, set) => {
      const el = $(name);
      el.innerHTML = opts.map(([k, t]) => `<button type="button" data-k="${k}" aria-pressed="${get() === k}">${t}</button>`).join("");
      el.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; set(b.dataset.k); [...el.children].forEach((c) => c.setAttribute("aria-pressed", c.dataset.k === get())); draw(); });
    };
    seg("#route-fam", Object.entries(FAM_NAME), () => fam, (k) => (fam = k));
    seg("#route-metric", [["acc", "All questions"], ["l600", "New-product questions, first 600 of the launch"]], () => metric, (k) => (metric = k));
    seg("#route-show", [["two", "Old rule vs learned rule"], ["all", "All four rules"]], () => show, (k) => (show = k));
    draw();
  }

  /* ---------- see it run: tabs + live service ---------- */
  const SERVICE = "https://guarded-distillation-triage.onrender.com";
  const pct = (x, d = 0) => (x == null ? "n/a" : (x * 100).toFixed(d) + "%");
  function explore() {
    const tabs = [["#t-replay", "#p-replay", "#f-replay"], ["#t-clinc", "#p-clinc", "#f-clinc"], ["#t-live", "#p-live", null]];
    let timer = null;
    function select(i, push) {
      tabs.forEach(([t, p], k) => { $(t).setAttribute("aria-selected", k === i); $(t).tabIndex = k === i ? 0 : -1; $(p).hidden = k !== i; });
      const f = tabs[i][2] && $(tabs[i][2]);
      if (f && !f.getAttribute("src")) f.setAttribute("src", f.dataset.src);
      clearInterval(timer);
      if (i === 2) { poll(); timer = setInterval(poll, 10000); }
    }
    tabs.forEach(([t], i) => {
      $(t).addEventListener("click", () => select(i));
      $(t).addEventListener("keydown", (e) => { if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const n = (i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length; select(n); $(tabs[n][0]).focus(); } });
    });
    ["#f-replay", "#f-clinc"].forEach((id) => {
      const frame = $(id);
      frame.addEventListener("load", () => { const fit = () => { try { frame.style.height = frame.contentDocument.documentElement.scrollHeight + 8 + "px"; } catch (e) {} }; fit(); setInterval(fit, 1500); });
    });
    const card = (v, label, cls = "", extra = "") => `<div class="card ${cls}"><b>${v}</b><span>${label}</span>${extra}</div>`;
    async function poll() {
      const banner = $("#live-banner");
      try {
        const [m, a] = await Promise.all([fetch(SERVICE + "/metrics").then((r) => r.json()), fetch(SERVICE + "/alerts").then((r) => r.json())]);
        const over = m.leak_over_budget, leak = m.estimated_leak_rate, bud = m.leak_budget, alerts = a.alerts || [];
        banner.className = "banner " + (alerts.length ? "bad" : "ok");
        banner.textContent = alerts.length ? "Needs a look: " + alerts.map((x) => x.detail).join("; ") : "Healthy. No stress spike, leaks within budget, spend under the cap.";
        $("#live-grid").innerHTML =
          card(m.model_version, "model version in use") + card(m.decisions_total.toLocaleString(), "questions routed so far") +
          card(pct(m.escalation_rate), `sent to Claude, last ${m.window} questions`) +
          card(pct(leak, 1), `estimated wrong answers that slipped through, against a budget of ${pct(bud, 1)}`, over ? "bad" : "",
               `<div class="meter-bar"><i class="${over ? "over" : ""}" style="width:${Math.min(100, ((leak || 0) / bud) * 100)}%"></i></div>`) +
          card(pct(m.audit.rate, 1), `of ${m.audit.audited} audited answers disagreed with Claude`) +
          card("$" + m.spend.today_usd.toFixed(3), `spent today of a $${m.spend.cap_usd.toFixed(2)} cap (${m.spend.cache_hits} cached, ${m.spend.api_calls} live calls)`) +
          card(m.promotions + " / " + m.rollbacks, "model promotions / automatic rollbacks");
        const top = m.stress.top || [];
        $("#live-stress").innerHTML = top.length
          ? top.map((t) => `<div class="stress-row"><span>${t.intent}</span><div class="tr"><i class="${t.stress >= m.stress.spike_threshold ? "hot" : ""}" style="width:${Math.min(100, t.stress * 100)}%"></i></div><span>${t.stress.toFixed(2)}</span></div>`).join("")
          : '<p class="small">No intent is under stress.</p>';
        $("#live-foot").textContent = `Uptime ${Math.round(m.uptime_s / 60)} min. ${m.spend.persistent ? "The log survives restarts." : "The free server's log resets when it restarts."} Updated ${new Date().toLocaleTimeString()}.`;
      } catch (e) {
        banner.className = "banner warn";
        banner.textContent = "Waiting for the server. A free instance sleeps when idle; the first request can take up to a minute. Retrying every 10 seconds.";
      }
    }
    $("#try-go").addEventListener("click", async () => {
      const out = $("#try-out"); out.textContent = "working...";
      try {
        const r = await fetch(SERVICE + "/triage", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text: $("#try-q").value }) });
        const j = await r.json();
        out.textContent = j.intent ? `${j.intent}  (${j.decision === "accept" ? "answered by the small model" : "sent to Claude"}, team: ${j.team}, urgency: ${j.urgency})` : JSON.stringify(j);
      } catch (e) { out.textContent = "The server did not answer. It may be waking up; try again in a minute."; }
      poll();
    });
    select(0);
  }

  /* ---------- nav highlight ---------- */
  function spy() {
    const links = [...document.querySelectorAll(".top nav a[href^='#']")];
    links.forEach((a) => {
      const sec = document.getElementById(a.getAttribute("href").slice(1));
      if (!sec) return;
      new IntersectionObserver((es) => es.forEach((e) => {
        if (e.isIntersecting) { links.forEach((l) => l.classList.remove("on")); a.classList.add("on"); }
      }), { rootMargin: "-45% 0px -50% 0px" }).observe(sec);
    });
  }

  calc();
  story();
  bars();
  dial();
  routing();
  explore();
  spy();
})();
