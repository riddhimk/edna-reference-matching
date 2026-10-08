/* eDNA-TaxAI motion layer -- plain JavaScript, loaded before React so the
   loading screen paints immediately.

   Everything here is framework-agnostic and works by event delegation on the
   document, so the React views stay free of animation plumbing:

     * Boot loader   a rotating 2.5D DNA helix shown only while the app is
                     genuinely initialising (CDN libraries downloading, JSX
                     compiling in the browser, first render). Its progress line
                     reflects those real stages, never a timer. On ready, the
                     same helix decelerates into the logo's resting pose and
                     flies into the sidebar brand mark, then hands over to it.
     * Cursor        a responsive dot with glowing aura (fine pointers only).
     * Magnetic      buttons drift a few pixels toward the cursor.
     * Tilt/glow     small cards tilt slightly; every card carries a soft
                     highlight that follows the pointer.
     * Ambient       a faint particle field behind the app, 30 fps, paused
                     when the tab is hidden, absent on touch/reduced motion.
     * Reveal        cards fade/rise in once as they enter the viewport.

   Respects prefers-reduced-motion throughout: the loader becomes a static
   helix with a cross-fade, and pointer effects and particles are off. */
(function () {
  "use strict";
  const doc = document.documentElement;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finePointer = matchMedia("(pointer: fine)").matches && matchMedia("(hover: hover)").matches;
  const mobile = matchMedia("(max-width: 860px)").matches;
  doc.classList.add(reduced ? "fx-still" : "fx-motion");
  if (finePointer && !reduced) doc.classList.add("fx-pointer");

  const FX = (window.EDNAFX = { reduced, finePointer });
  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  /* ---------------------------------------------------------- helix ---- */

  /* One geometry for the loader and the logo, so the loader can become the
     logo: `turns` full twists over the height, `pairs` base pairs. */
  const TURNS = 2, PAIRS = 22;
  FX.TURNS = TURNS; FX.PAIRS = PAIRS;

  /* SVG paths of the helix at rest (front view, rotation 0): the shape the
     loader settles into and the sidebar logo draws. */
  FX.helixPaths = function (w, h, pad) {
    pad = pad == null ? 1.5 : pad;
    const R = w / 2 - pad, cx = w / 2, n = 64, a = [], b = [], rungs = [];
    for (let i = 0; i <= n; i++) {
      const t = i / n, y = pad + t * (h - 2 * pad), th = t * TURNS * Math.PI * 2;
      a.push(`${i ? "L" : "M"}${(cx + R * Math.cos(th)).toFixed(2)},${y.toFixed(2)}`);
      b.push(`${i ? "L" : "M"}${(cx - R * Math.cos(th)).toFixed(2)},${y.toFixed(2)}`);
    }
    const nr = Math.round(TURNS * 4) + 1;
    for (let k = 0; k < nr; k++) {
      const t = (k + 0.5) / nr, y = pad + t * (h - 2 * pad), th = t * TURNS * Math.PI * 2;
      const x1 = cx + R * Math.cos(th), x2 = cx - R * Math.cos(th);
      if (Math.abs(x1 - x2) > R * 0.5) rungs.push(`M${x1.toFixed(2)},${y.toFixed(2)}L${x2.toFixed(2)},${y.toFixed(2)}`);
    }
    return { a: a.join(""), b: b.join(""), rungs: rungs.join("") };
  };

  /* Draws the helix into a 2D context. rot = rotation about the vertical
     axis; tilt = slight lean; alpha = global fade. Back-to-front so nearer
     nodes overlap farther ones. */
  function drawHelix(ctx, W, H, rot, opts) {
    const o = Object.assign({ R: W * 0.22, top: H * 0.1, bottom: H * 0.9, alpha: 1, glow: 1, tilt: 0 }, opts);
    const cx = W / 2, f = o.R * 6;
    const items = [];
    const project = (t, phase) => {
      const th = t * TURNS * Math.PI * 2 + rot + phase;
      const x = o.R * Math.cos(th), z = o.R * Math.sin(th);
      const s = f / (f - z);
      const y = lerp(o.top, o.bottom, t);
      return { x: cx + x * s + o.tilt * (y - H / 2), y, z, s };
    };
    // backbones as many short segments, depth-shaded
    const seg = 90;
    for (const phase of [0, Math.PI]) {
      let prev = project(0, phase);
      for (let i = 1; i <= seg; i++) {
        const p = project(i / seg, phase);
        items.push({ z: (p.z + prev.z) / 2, k: "seg", a: prev, b: p });
        prev = p;
      }
    }
    for (let i = 0; i < PAIRS; i++) {
      const t = (i + 0.5) / PAIRS;
      const p = project(t, 0), q = project(t, Math.PI);
      items.push({ z: (p.z + q.z) / 2 - 0.01, k: "rung", a: p, b: q });
      items.push({ z: p.z, k: "node", a: p });
      items.push({ z: q.z, k: "node", a: q });
    }
    items.sort((m, n) => m.z - n.z);
    for (const it of items) {
      const depth = clamp((it.z / o.R + 1) / 2, 0, 1);           // 0 far .. 1 near
      const al = o.alpha * (0.22 + 0.78 * depth);
      if (it.k === "seg") {
        ctx.strokeStyle = `rgba(190,214,255,${al * 0.9})`;
        ctx.lineWidth = 1.2 + 1.6 * depth;
        ctx.beginPath(); ctx.moveTo(it.a.x, it.a.y); ctx.lineTo(it.b.x, it.b.y); ctx.stroke();
      } else if (it.k === "rung") {
        ctx.strokeStyle = `rgba(150,190,255,${al * 0.38})`;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(it.a.x, it.a.y); ctx.lineTo(it.b.x, it.b.y); ctx.stroke();
      } else {
        const r = (2.1 + 2.2 * depth) * it.a.s;
        if (o.glow > 0) {
          ctx.globalAlpha = al * 0.32 * o.glow;
          ctx.drawImage(GLOW, it.a.x - r * 4, it.a.y - r * 4, r * 8, r * 8);
          ctx.globalAlpha = 1;
        }
        ctx.fillStyle = `rgba(226,238,255,${al})`;
        ctx.beginPath(); ctx.arc(it.a.x, it.a.y, r, 0, Math.PI * 2); ctx.fill();
      }
    }
  }

  // a soft glow sprite, drawn once (cheaper than shadowBlur every frame)
  const GLOW = document.createElement("canvas");
  GLOW.width = GLOW.height = 64;
  (function () {
    const g = GLOW.getContext("2d"), grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, "rgba(140,190,255,1)"); grd.addColorStop(0.35, "rgba(110,160,255,.35)"); grd.addColorStop(1, "rgba(90,140,255,0)");
    g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
  })();

  /* --------------------------------------------------------- loader ---- */

  const STAGES = { libraries: [0.18, "Loading libraries"], compiling: [0.55, "Compiling interface"],
                   views: [0.8, "Preparing views"], ready: [1, "Ready"] };
  let boot = null;

  function buildLoader() {
    const el = document.getElementById("boot");
    if (!el) return;
    const stage = el.querySelector(".boot-stage");
    const canvas = el.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    const dpr = Math.min(2, devicePixelRatio || 1);
    const CW = 260, CH = 420;
    canvas.width = CW * dpr; canvas.height = CH * dpr;
    canvas.style.width = CW + "px"; canvas.style.height = CH + "px";
    ctx.scale(dpr, dpr);

    // molecular particles around the helix
    const parts = Array.from({ length: mobile ? 26 : 46 }, () => ({
      x: Math.random() * CW, y: Math.random() * CH, z: Math.random(),
      vx: (Math.random() - 0.5) * 0.12, vy: (Math.random() - 0.5) * 0.12 }));

    boot = { el, stage, canvas, ctx, CW, CH, parts, rot: 0, speed: 0.9, settle: null, alpha: 1, mx: 0, my: 0, px: 0, py: 0,
             t0: performance.now(), raf: 0, done: false };
    FX.stage("libraries");
    if (!reduced) {
      addEventListener("pointermove", onBootMove, { passive: true });
      boot.raf = requestAnimationFrame(bootFrame);
    } else {
      drawStill();
    }
    // the libraries stage ends once React has arrived
    (function waitReact() {
      if (!boot || boot.done) return;
      if (window.React && window.ReactDOM && window.Babel) FX.stage("compiling");
      else setTimeout(waitReact, 60);
    })();
  }

  function onBootMove(e) {
    if (!boot) return;
    boot.mx = (e.clientX / innerWidth - 0.5); boot.my = (e.clientY / innerHeight - 0.5);
  }

  function drawStill() {
    const b = boot;
    b.ctx.clearRect(0, 0, b.CW, b.CH);
    drawHelix(b.ctx, b.CW, b.CH, 0, { R: b.CW * 0.2, glow: 0.6 });
  }

  let lastT = performance.now();
  function bootFrame(now) {
    const b = boot;
    if (!b) return;
    const dt = Math.min(50, now - lastT) / 1000; lastT = now;
    if (document.hidden) { b.raf = requestAnimationFrame(bootFrame); return; }
    // physically weighted rotation: settles onto the rest pose when asked
    if (b.settle) {
      const target = b.settle.target;
      b.rot = lerp(b.rot, target, 1 - Math.pow(0.0009, dt));
      b.alphaParts = Math.max(0, (b.alphaParts == null ? 1 : b.alphaParts) - dt * 2.2);
    } else {
      b.rot += b.speed * dt;
    }
    b.px = lerp(b.px, b.mx, 0.04); b.py = lerp(b.py, b.my, 0.04);
    b.el.style.setProperty("--px", (b.px * 18).toFixed(2) + "px");
    b.el.style.setProperty("--py", (b.py * 18).toFixed(2) + "px");

    const { ctx, CW, CH } = b;
    ctx.clearRect(0, 0, CW, CH);
    const pa = b.alphaParts == null ? 1 : b.alphaParts;
    if (pa > 0) {
      for (const p of b.parts) {
        p.x += p.vx + b.px * 0.05 * p.z; p.y += p.vy + b.py * 0.05 * p.z;
        if (p.x < -10) p.x = CW + 10; if (p.x > CW + 10) p.x = -10;
        if (p.y < -10) p.y = CH + 10; if (p.y > CH + 10) p.y = -10;
      }
      ctx.lineWidth = 0.6;
      for (let i = 0; i < b.parts.length; i++) {
        const p = b.parts[i];
        for (let j = i + 1; j < b.parts.length; j++) {
          const q = b.parts[j], d = Math.hypot(p.x - q.x, p.y - q.y);
          if (d < 46) { ctx.strokeStyle = `rgba(140,180,255,${0.12 * (1 - d / 46) * pa})`; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke(); }
        }
        ctx.fillStyle = `rgba(170,200,255,${(0.15 + 0.35 * p.z) * pa})`;
        ctx.beginPath(); ctx.arc(p.x, p.y, 0.6 + p.z * 1.1, 0, Math.PI * 2); ctx.fill();
      }
    }
    drawHelix(ctx, CW, CH, b.rot, { R: CW * 0.2, glow: b.settle ? Math.max(0.25, 1 - (now - b.settle.t) / 900) : 1,
                                    tilt: b.settle ? 0 : 0.06 + b.px * 0.05 });
    b.raf = requestAnimationFrame(bootFrame);
  }

  FX.stage = function (name) {
    if (!boot || boot.done) return;
    const s = STAGES[name];
    if (!s) return;
    const cur = parseFloat(boot.el.style.getPropertyValue("--progress") || 0);
    if (s[0] <= cur) return;                         // stages only move forward
    boot.el.style.setProperty("--progress", s[0]);
    boot.stage.textContent = s[1];
  };

  /* Called by the app once its first view has rendered. The helix settles
     into the logo's rest pose, then flies into the sidebar's brand mark. */
  FX.ready = function () {
    if (!boot || boot.done) return;
    FX.stage("ready");
    boot.done = true;
    const b = boot;
    // fly to the brand mark that is actually on screen (sidebar, or the mobile bar)
    const logos = [...document.querySelectorAll("[data-brand-logo]")];
    const logo = logos.find((l) => { const r = l.getBoundingClientRect(); return r.width > 0 && r.right > 0 && r.left < innerWidth && r.bottom > 0 && r.top < innerHeight; });
    const land = () => logos.forEach((l) => l.classList.add("is-landed"));
    doc.classList.add("fx-booted");
    const finish = () => {
      cancelAnimationFrame(b.raf);
      removeEventListener("pointermove", onBootMove);
      land();
      b.el.remove();
      boot = null;
      doc.classList.add("fx-live");
    };
    if (reduced || !logo || !b.canvas.getBoundingClientRect().width) {
      land();
      b.el.classList.add("is-leaving");
      setTimeout(finish, 320);
      return;
    }
    // 1. decelerate onto the nearest rest pose (rotation multiple of 2*pi)
    b.settle = { target: Math.ceil(b.rot / (Math.PI * 2) + 0.15) * Math.PI * 2, t: performance.now() };
    // 2. after a short settle, fly the canvas onto the logo
    setTimeout(() => {
      const from = b.canvas.getBoundingClientRect();
      const to = logo.getBoundingClientRect();
      // the helix occupies the middle band of the canvas: match its drawn box to the logo box
      const helixW = b.CW * 0.4 + 9, helixH = b.CH * 0.8;
      const scale = Math.min(to.width / helixW, to.height / helixH);
      const fromCx = from.left + from.width / 2, fromCy = from.top + from.height / 2;
      const dx = to.left + to.width / 2 - fromCx, dy = to.top + to.height / 2 - fromCy;
      b.el.classList.add("is-leaving");
      const anim = b.canvas.animate(
        [{ transform: "translate(0,0) scale(1)" },
         { transform: `translate(${dx}px, ${dy}px) scale(${scale})` }],
        { duration: 1100, easing: "cubic-bezier(.22, 1, .36, 1)", fill: "forwards" });
      setTimeout(land, 820);                                    // the real logo takes over in flight
      anim.onfinish = () => {
        b.canvas.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 160, fill: "forwards" }).onfinish = finish;
      };
    }, 260);
  };

  /* ------------------------------------------------------- ambient ---- */

  function ambient() {
    // Disabled to prevent infinite 30fps style invalidation and background repaints
  }

  /* -------------------------------------------------------- cursor ---- */

  function cursor() {
    if (!finePointer || reduced) return;
    const dot = document.createElement("div");
    dot.className = "fx-cursor-dot";
    dot.setAttribute("aria-hidden", "true");
    document.body.append(dot);
    let shown = false, lastHot = false, lastText = false, checkPending = false;
    addEventListener("pointermove", (e) => {
      if (e.pointerType !== "mouse") return;
      dot.style.transform = `translate3d(${e.clientX}px, ${e.clientY}px, 0)`;
      if (!shown) { shown = true; doc.classList.add("fx-cursor-on"); }
      if (!checkPending) {
        checkPending = true;
        requestAnimationFrame(() => {
          checkPending = false;
          const t = e.target;
          if (!t) return;
          const text = !!(t.closest && t.closest("input, textarea, select, [contenteditable]"));
          const hot = !text && !!(t.closest && t.closest("a, button, [role=tab], label, summary, tr.click, .navlink, .marker, .suggest button"));
          if (hot !== lastHot) { lastHot = hot; doc.classList.toggle("fx-cursor-hot", hot); }
          if (text !== lastText) { lastText = text; doc.classList.toggle("fx-cursor-text", text); }
        });
      }
    }, { passive: true });
    addEventListener("pointerdown", () => doc.classList.add("fx-cursor-down"));
    addEventListener("pointerup", () => doc.classList.remove("fx-cursor-down"));
    document.addEventListener("mouseleave", () => { shown = false; doc.classList.remove("fx-cursor-on"); });
  }

  /* --------------------------------------------- magnetic + tilt ---- */

  function tactile() {
    // Disabled to eliminate getBoundingClientRect layout reflows and 3D tilt overhead
  }

  /* -------------------------------------------------------- reveal ---- */

  function reveal() {
    // Instant reveal: elements appear immediately without mutation observers or delay
    const els = document.querySelectorAll(".card, .tile, .hero, .table-wrap");
    for (const el of els) el.classList.add("fx-in");
  }

  /* ---------------------------------------------------------- boot ---- */

  buildLoader();
  // Safety nets: an error during start-up is shown on the loader rather than
  // leaving it spinning, and the app is never kept hidden indefinitely.
  addEventListener("error", (e) => {
    if (!boot || boot.done) return;
    boot.stage.textContent = "Something went wrong while starting — see the browser console";
    boot.el.classList.add("is-error");
  });
  setTimeout(() => { if (boot && !boot.done) FX.ready(); }, 20000);
  const start = () => { ambient(); cursor(); tactile(); reveal(); };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
