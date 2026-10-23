"""The online gate: scope gate, intent gate, escalation, audits, per-intent stress (monitor only)."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np

from ..gate import accepts, runner_up
from ..semantic import IntentGraph, SemanticField
from ..taxonomy import OOS, route, urgency
from .bundle import IN, Bundle
from .store import Store
from .teacher_access import TeacherAccess, TeacherUnavailable

AUDIT_P = 0.08


class Triage:
    def __init__(self, bundle: Bundle, teacher: TeacherAccess, store: Store, audit_p: float = AUDIT_P, seed: int = 0):
        self.teacher, self.store, self.audit_p = teacher, store, audit_p
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()
        self.prev: dict | None = None          # {"bundle", "rate", "audited"} of the version before the current one
        self.blocked: set[str] = set()         # versions rolled back: never promoted again
        self.n_cur, self.checked = 0, False
        self.set_bundle(bundle)
        self.started = time.time()

    def set_bundle(self, bundle: Bundle) -> None:
        self.b = bundle
        self.idx = {n: i for i, n in enumerate(bundle.intents)}
        cfg = SimpleNamespace(sem_gain=0.7, sem_top=3, sem_spread=0.5, sem_hot=0.4, sem_decay=0.002, sem_release=30)
        g = IntentGraph(bundle.intents)
        g.rebuild({}, [])
        self.stress = SemanticField(g, cfg)        # observed stress: monitors the intents, does not change decisions

    def decide(self, text: str) -> dict:
        t0 = time.time()
        b = self.b
        P_s = b.scope.proba([text])
        u_s = float(b.scale_scope(runner_up(P_s))[0])
        scope_top = IN if P_s[0, b.scope.index[IN]] >= P_s[0, b.scope.index[OOS]] else OOS
        scope_ok = bool(accepts(u_s, b.q0_scope))
        P_i = b.intent.proba([text])
        u_i = float(b.scale_intent(runner_up(P_i))[0])
        intent_top = b.intent.universe[int(P_i[0].argmax())]
        intent_ran = scope_ok and scope_top == IN
        rec = {"text": text, "u_scope": round(u_s, 4), "u_intent": round(u_i, 4), "student_intent": intent_top, "student_scope": scope_top}
        usd, degraded = 0.0, None
        if scope_ok and scope_top == OOS:
            decision, answer = "accept", OOS
        elif intent_ran and accepts(u_i, b.q0_intent):
            decision, answer = "accept", intent_top
        else:
            decision, answer = "escalate", None
        audited, teacher_label, source = False, None, "student"
        if decision == "escalate":
            try:
                teacher_label, source, usd = self.teacher.label(text)
                answer = teacher_label
            except TeacherUnavailable as e:
                degraded = str(e)
                answer = OOS if scope_top == OOS and scope_ok else intent_top      # best effort from the student
                source = "student_degraded"
        elif self.rng.random() < self.audit_p:
            audited = True
            try:
                teacher_label, source, usd = self.teacher.label(text)
            except TeacherUnavailable:
                audited = False
        disagree = bool(audited and teacher_label is not None and teacher_label != answer)
        with self.lock:
            if teacher_label in self.idx and intent_top in self.idx:        # teacher-seen answer: update observed stress
                wrong = float(intent_top != teacher_label)
                k = self.idx[intent_top]
                self.stress.observe(k, wrong == 0.0)
                self.stress.inject(k, 0.15 * (wrong - float(b.calib_intent(u_i))))
            self.stress.tick()
        rec.update(decision=decision, audited=audited, answer=answer, teacher_label=teacher_label, source=source, disagree=disagree,
                   usd=round(usd, 7), p_audit=self.audit_p if decision == "accept" else None, latency_ms=round((time.time() - t0) * 1000, 1))
        if degraded:
            rec["degraded"] = degraded
        self.store.add_decision(b.version, rec)
        self.n_cur += 1
        from .rollout import maybe_rollback
        maybe_rollback(self)
        out = {k: rec[k] for k in ("decision", "audited", "source", "u_intent", "latency_ms")}
        out.update(intent=answer, team=route(answer) if answer else None, urgency=urgency(answer) if answer else None,
                   model_version=b.version, cost_usd=rec["usd"])
        if degraded:
            out["degraded"] = degraded
        return out
