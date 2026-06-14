/* ============================================================================
   Visual enhancement layer — ambient canvas, reveals, count-ups, tooltips,
   signal indicator, theme toggle. Everything is feature-detected and gated
   behind prefers-reduced-motion; the page is fully usable with none of it.
   ========================================================================== */

import { traceTeam, computeFlipTransforms } from "./trace.js";

const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;

let ambientStarted = false;
let toolingStarted = false;

/* ---------- public ---------- */
export function initChrome() {
  if (toolingStarted) return;
  toolingStarted = true;
  try { initTheme(); } catch (e) { /* non-fatal */ }
  try { initAmbient(); } catch (e) { /* non-fatal */ }
}

export function afterRender(root, { animate = true } = {}) {
  if (!root) return;
  try {
    if (animate && !REDUCE) {
      revealStagger(root);
      countUps(root);
    } else {
      root.querySelectorAll(".reveal").forEach((el) => el.classList.add("is-in"));
      root.querySelectorAll("[data-count]").forEach((el) => {
        const t = Number(el.dataset.count);
        if (Number.isFinite(t)) el.textContent = String(t);
      });
    }
    bindTooltips(root);
  } catch (e) {
    // effects must never break the core render
    root.querySelectorAll(".reveal").forEach((el) => el.classList.add("is-in"));
  }
}

export function setSignal(sourceStatus = {}, latest = {}) {
  const el = document.getElementById("signal");
  if (!el) return;
  const status = latest.source_status;
  const cls = (status === "cached" || status === "fallback_success")
    ? "is-warn"
    : status === "failed" ? "is-bad" : "";
  el.className = "signal" + (cls ? " " + cls : "");
  const txt = el.querySelector(".signal__text");
  if (txt) {
    const label = sourceStatus.visible_status || status || "ONLINE";
    txt.textContent = "FEED · " + label;
  }
}

/* ---------- reveals ---------- */
function revealStagger(root) {
  const items = root.querySelectorAll(".reveal");
  const io = "IntersectionObserver" in window
    ? new IntersectionObserver((entries, obs) => {
        entries.forEach((e) => {
          if (e.isIntersecting) { e.target.classList.add("is-in"); obs.unobserve(e.target); }
        });
      }, { threshold: 0.12 })
    : null;
  items.forEach((el) => {
    const rect = el.getBoundingClientRect();
    if (!io || rect.top < window.innerHeight * 0.94) {
      requestAnimationFrame(() => el.classList.add("is-in"));
    } else {
      io.observe(el);
    }
  });
}

/* ---------- count-ups (GSAP if present, rAF fallback) ---------- */
function countUps(root) {
  root.querySelectorAll("[data-count]").forEach((el) => {
    const target = Number(el.dataset.count);
    if (!Number.isFinite(target)) return;
    const flash = () => {
      el.classList.add("count-flash");
      setTimeout(() => el.classList.remove("count-flash"), 600);
    };
    if (target <= 0) { el.textContent = "0"; return; }

    if (window.gsap) {
      const proxy = { v: 0 };
      window.gsap.to(proxy, {
        v: target, duration: 1.4, ease: "expo.out",
        onUpdate: () => { el.textContent = String(Math.round(proxy.v)); },
        onComplete: () => { el.textContent = String(target); flash(); }
      });
      return;
    }
    // rAF fallback
    const dur = 1300;
    let start = null;
    const ease = (t) => (t === 1 ? 1 : 1 - Math.pow(2, -10 * t));
    const step = (ts) => {
      if (start === null) start = ts;
      const t = Math.min(1, (ts - start) / dur);
      el.textContent = String(Math.round(target * ease(t)));
      if (t < 1) requestAnimationFrame(step);
      else flash();
    };
    requestAnimationFrame(step);
  });
}

/* ---------- tooltips ---------- */
function bindTooltips(root) {
  const tip = document.getElementById("tip");
  if (!tip) return;
  const show = (el) => {
    tip.innerHTML = el.getAttribute("data-tip") || "";
    tip.classList.add("is-on");
    tip.setAttribute("aria-hidden", "false");
  };
  const hide = () => { tip.classList.remove("is-on"); tip.setAttribute("aria-hidden", "true"); };
  const place = (x, y) => {
    const pad = 14;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let left = x + 16, top = y + 16;
    if (left + w + pad > window.innerWidth) left = x - w - 16;
    if (top + h + pad > window.innerHeight) top = y - h - 16;
    tip.style.left = Math.max(pad, left) + "px";
    tip.style.top = Math.max(pad, top) + "px";
  };
  root.querySelectorAll("[data-tip]").forEach((el) => {
    el.addEventListener("mouseenter", (e) => { show(el); place(e.clientX, e.clientY); });
    el.addEventListener("mousemove", (e) => place(e.clientX, e.clientY));
    el.addEventListener("mouseleave", hide);
    el.addEventListener("focus", () => {
      const r = el.getBoundingClientRect();
      show(el); place(r.left, r.bottom);
    });
    el.addEventListener("blur", hide);
  });
}

/* ---------- theme toggle (ember <-> arctic) ---------- */
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("majors-theme"); } catch (_) {}
  if (saved === "arctic") document.body.classList.add("theme-arctic");
  document.querySelectorAll('[data-action="theme"]').forEach((btn) => {
    btn.addEventListener("click", () => {
      const arctic = document.body.classList.toggle("theme-arctic");
      try { localStorage.setItem("majors-theme", arctic ? "arctic" : "ember"); } catch (_) {}
    });
  });
}

/* ---------- ambient tactical canvas ---------- */
function initAmbient() {
  if (ambientStarted) return;
  const canvas = document.getElementById("ambient");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ambientStarted = true;

  let w = 0, h = 0, dpr = Math.min(window.devicePixelRatio || 1, 2);
  let particles = [];

  function resize() {
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const count = Math.min(64, Math.round((w * h) / 26000));
    particles = Array.from({ length: count }, () => spawn());
  }

  function spawn() {
    const cyan = Math.random() < 0.32;
    return {
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.18,
      vy: -0.08 - Math.random() * 0.18,
      r: 0.6 + Math.random() * 1.4,
      cyan
    };
  }

  function frame() {
    ctx.clearRect(0, 0, w, h);
    const D = 132;
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.x += p.vx; p.y += p.vy;
      if (p.y < -10) { p.y = h + 10; p.x = Math.random() * w; }
      if (p.x < -10) p.x = w + 10;
      if (p.x > w + 10) p.x = -10;
      // links
      for (let j = i + 1; j < particles.length; j++) {
        const q = particles[j];
        const dx = p.x - q.x, dy = p.y - q.y;
        const dist = Math.hypot(dx, dy);
        if (dist < D) {
          const a = (1 - dist / D) * 0.14;
          ctx.strokeStyle = `rgba(255,138,46,${a})`;
          ctx.lineWidth = 0.6;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(q.x, q.y);
          ctx.stroke();
        }
      }
      // dot
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.cyan ? "rgba(70,210,255,0.5)" : "rgba(255,150,70,0.5)";
      ctx.fill();
    }
    raf = requestAnimationFrame(frame);
  }

  let raf = null;
  function start() { if (!raf && !document.hidden) raf = requestAnimationFrame(frame); }
  function stop() { if (raf) { cancelAnimationFrame(raf); raf = null; } }

  function drawStatic() {
    ctx.clearRect(0, 0, w, h);
    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.cyan ? "rgba(70,210,255,0.35)" : "rgba(255,150,70,0.35)";
      ctx.fill();
    }
  }

  resize();

  if (REDUCE) {
    // single static frame, repaint on resize, no animation loop
    drawStatic();
    window.addEventListener("resize", debounce(() => { resize(); drawStatic(); }, 200));
    return;
  }

  window.addEventListener("resize", debounce(resize, 200));
  document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));
  start();
}

function debounce(fn, ms) {
  let t = null;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

/* ---------- PREDICT: hover-to-trace a team's whole path ---------- */
export function initPredictTrace(root, bracket) {
  if (!root) return;
  const board = root.querySelector(".predict-board");
  if (!board || !bracket) return;
  const clear = () => {
    board.classList.remove("has-trace");
    board.querySelectorAll(".is-traced").forEach((el) => el.classList.remove("is-traced"));
  };
  const apply = (team) => {
    if (!team) { clear(); return; }
    const { matchKeys } = traceTeam(bracket, team);
    board.classList.add("has-trace");
    board.querySelectorAll("[data-mk]").forEach((card) => {
      card.classList.toggle("is-traced", matchKeys.has(card.dataset.mk));
    });
  };
  const teamOf = (e) => {
    const el = e.target.closest && e.target.closest("[data-predict-team]");
    return el ? el.dataset.predictTeam : null;
  };
  board.addEventListener("mouseover", (e) => apply(teamOf(e)));
  board.addEventListener("mouseleave", clear);
  board.addEventListener("focusin", (e) => apply(teamOf(e)));
  board.addEventListener("focusout", clear);
}

/* ---------- PREDICT: FLIP reorder animation ---------- */
export function captureRects(root) {
  const rects = {};
  if (!root) return rects;
  root.querySelectorAll("[data-mk]").forEach((el) => {
    const r = el.getBoundingClientRect();
    rects[el.dataset.mk] = { left: r.left, top: r.top };
  });
  return rects;
}

export function playFlip(root, oldRects) {
  if (!root || REDUCE || !window.gsap || !oldRects) return;
  const moves = computeFlipTransforms(oldRects, captureRects(root));
  if (!moves.length) return;
  const byKey = new Map();
  root.querySelectorAll("[data-mk]").forEach((el) => byKey.set(el.dataset.mk, el));
  for (const { key, dx, dy } of moves) {
    const el = byKey.get(key);
    if (el) window.gsap.fromTo(el, { x: dx, y: dy }, { x: 0, y: 0, duration: 0.42, ease: "power2.out" });
  }
}

/* ---------- PREDICT: non-silent conflict toast ---------- */
export function showConflictToast(dropped) {
  if (!dropped || !dropped.length) return;
  let toast = document.querySelector(".predict-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "predict-toast";
    toast.setAttribute("role", "status");
    toast.setAttribute("aria-live", "polite");
    document.body.appendChild(toast);
  }
  const labels = dropped.slice(0, 3).map((d) => d.label);
  const more = dropped.length > 3 ? ` 等 ${dropped.length} 项` : "";
  toast.textContent = `重排后这些预测已失效：${labels.join("、")}${more}`;
  toast.classList.add("is-on");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => toast.classList.remove("is-on"), 2200);
}
