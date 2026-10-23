"""Rolling service metrics from the decision store."""
from __future__ import annotations

import numpy as np

SPIKE = 0.6        # an intent's observed stress at or above this is a spike (alerts)


def audit_stats(rows: list[dict]) -> dict:
    aud = [r for r in rows if r.get("audited") and r.get("teacher_label") is not None]
    dis = [r for r in aud if r.get("disagree")]
    return {"audited": len(aud), "disagreements": len(dis), "rate": (len(dis) / len(aud)) if aud else None}


def compute(triage, window: int = 500) -> dict:
    st, b = triage.store, triage.b
    rows = st.last(window)
    n = len(rows)
    esc = sum(r["decision"] == "escalate" for r in rows)
    leak = sum(1.0 / r["p_audit"] for r in rows if r.get("disagree") and r.get("p_audit")) / n if n else 0.0
    budget = b.alpha_scope + b.alpha_intent
    s = triage.stress.stress
    top = [(b.intents[i], float(s[i])) for i in np.argsort(-s)[:8] if s[i] > 0.01]
    promos = st.events("promoted", 20)
    return {
        "model_version": b.version,
        "decisions_total": st.count(),
        "window": n,
        "escalation_rate": (esc / n) if n else None,
        "estimated_leak_rate": leak,
        "leak_budget": budget,
        "leak_over_budget": bool(n >= 200 and leak > budget),
        "audit": audit_stats(rows),
        "stress": {"max": float(s.max()), "spike_threshold": SPIKE, "spike": bool(s.max() >= SPIKE),
                   "on_probation": int(triage.stress.on.sum()), "top": [{"intent": k, "stress": round(v, 3)} for k, v in top]},
        "promotions": len(promos), "recent_promotions": promos[:5],
        "rollbacks": len(st.events("rollback", 20)),
        "spend": {"today_usd": round(triage.teacher.spent_today(), 5), "cap_usd": triage.teacher.cap,
                  "total_usd": round(triage.teacher.ledger.total(), 5),
                  "cache_hits": triage.teacher.cache_hits, "api_calls": triage.teacher.api_calls},
        "uptime_s": int(__import__("time").time() - triage.started),
    }
