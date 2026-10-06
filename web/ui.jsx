/* Shared React pieces for the motion design system, used by app.jsx and
   analysis.jsx (loaded first, handed over through window.EdnaUI).

   Animation here is deliberately sparse: numbers count up once, charts draw
   in once when they enter the viewport, tooltips ease into place. Nothing
   loops. Everything collapses to its final state under reduced motion. */

const { useState, useEffect, useRef, useCallback } = React;
const FX = window.EDNAFX || {};
const REDUCED = !!FX.reduced;
if (FX.stage) FX.stage("views");

/* True once the element has meaningfully entered the viewport (then stays true). */
function useInView(options = {}) {
  const ref = useRef(null);
  const [seen, setSeen] = useState(REDUCED || !("IntersectionObserver" in window));
  useEffect(() => {
    if (seen || !ref.current) return;
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setSeen(true); io.disconnect(); } },
      { threshold: options.threshold ?? 0.2, rootMargin: options.rootMargin ?? "0px 0px -8% 0px" });
    io.observe(ref.current);
    return () => io.disconnect();
  }, [seen]);
  return [ref, seen];
}

/* Animates the number inside a display string from 0 when it becomes visible:
   "309,155" / "24.3%" / "12.0 min" / "+2" / "0.9668". Non-numeric strings
   ("—") render as they are. Formatting (thousands separators, decimals,
   prefix and suffix) is preserved on every frame. */
const NUM = /^([^\d\-−]*)([-−]?[\d,]*\.?\d+)(.*)$/s;
function CountUp({ value, duration = 1100 }) {
  const text = value === null || value === undefined ? "—" : String(value);
  const m = text.match(NUM);
  const [ref, seen] = useInView({ threshold: 0.3 });
  const [shown, setShown] = useState(m && !REDUCED ? null : text);
  useEffect(() => {
    if (!m) { setShown(text); return; }
    if (!seen) return;
    if (REDUCED) { setShown(text); return; }
    const [, pre, num, post] = m;
    const neg = /^[-−]/.test(num);
    const clean = num.replace(/[,−-]/g, "");
    const target = parseFloat(clean);
    const decimals = (clean.split(".")[1] || "").length;
    const commas = num.includes(",");
    const fmt = (v) => {
      let s = v.toFixed(decimals);
      if (commas) { const [i, d] = s.split("."); s = Number(i).toLocaleString("en-US") + (d ? "." + d : ""); }
      return pre + (neg ? "−" : "") + s + post;
    };
    let raf, t0;
    const step = (now) => {
      t0 = t0 || now;
      const p = Math.min(1, (now - t0) / duration);
      const e = 1 - Math.pow(1 - p, 4);                       // ease-out, settles gently
      setShown(p < 1 ? fmt(target * e) : text);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [seen, text]);
  // the final text reserves the width so the layout never jumps while counting
  return <span ref={ref} className="countup" data-final={text}>
    <span aria-hidden="true">{shown === null ? (m ? m[1] + "0" + (m[3] || "") : text) : shown}</span>
    <span className="sr-only">{text}</span>
  </span>;
}

/* Tooltips render into document.body: cards can be transformed (tilt, reveal),
   which would otherwise re-anchor fixed-position children to the card. */
function useTooltip() {
  const [tip, setTip] = useState(null);
  const show = useCallback((e, content) => setTip({ x: e.clientX, y: e.clientY, content }), []);
  const hide = useCallback(() => setTip(null), []);
  const node = tip ? ReactDOM.createPortal(
    <div className="tooltip" role="tooltip" style={{ left: tip.x, top: tip.y }}>{tip.content}</div>, document.body) : null;
  return { show, hide, node };
}

/* The brand mark: the same helix the loader draws, at rest. The loader flies
   into this element (data-brand-logo) and hands over to it. */
function HelixLogo({ width = 13, height = 36 }) {
  const p = FX.helixPaths ? FX.helixPaths(width, height, 1.2) : { a: "", b: "", rungs: "" };
  return <svg className="brand-logo" data-brand-logo width={width} height={height} viewBox={`0 0 ${width} ${height}`}
    fill="none" stroke="currentColor" strokeLinecap="round" aria-hidden="true">
    <path d={p.rungs} strokeWidth="0.8" opacity=".55" />
    <path d={p.b} strokeWidth="1.25" opacity=".7" />
    <path d={p.a} strokeWidth="1.4" />
  </svg>;
}

/* Capitalises the first letter of a sentence that may come from data. */
const cap = (s) => (s ? String(s)[0].toUpperCase() + String(s).slice(1) : s);

window.EdnaUI = { useInView, CountUp, useTooltip, HelixLogo, cap, REDUCED };
