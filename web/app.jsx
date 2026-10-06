/* eDNA-TaxAI results dashboard.
   React 18 via CDN, transpiled in-browser by Babel standalone -- this machine
   has no Node toolchain, so there is no build step. Charts are hand-rolled SVG
   following the project's data-viz mark specs (thin marks, rounded data-ends,
   recessive grid, hover tooltip, legend whenever two or more series appear). */

const { useState, useEffect, useMemo, useCallback, useRef, useLayoutEffect } = React;
const { CountUp, useTooltip, HelixLogo } = window.EdnaUI;      // web/ui.jsx

const API = "";                      // same origin -- the API serves this page
const SERIES = ["--s1","--s2","--s3","--s4","--s5","--s6","--s7","--s8"];
const cssVar = (v) => `var(${v})`;

/* ---------------------------------------------------------------- data --- */

/* `path` may be null, which means "nothing to fetch yet" -- the hook then stays
   idle rather than firing a placeholder request whose shape callers would have
   to defend against. */
function useApi(path, deps = []) {
  const [state, setState] = useState({ data: null, loading: !!path, error: null });
  useEffect(() => {
    if (!path) { setState({ data: null, loading: false, error: null }); return; }
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    fetch(API + path)
      .then(async (r) => {
        if (!r.ok) {
          let detail = `HTTP ${r.status}`;
          try { const j = await r.json(); if (j.detail) detail = j.detail; } catch (_) {}
          throw new Error(detail);
        }
        return r.json();
      })
      .then((d) => alive && setState({ data: d, loading: false, error: null }))
      .catch((e) => alive && setState({ data: null, loading: false, error: e.message }));
    return () => { alive = false; };
  }, deps);                                                    // eslint-disable-line
  return state;
}

const fmt = (n, d = 4) => (n === null || n === undefined ? "—" : Number(n).toFixed(d));
const pct = (n, d = 1) => (n === null || n === undefined ? "—" : (Number(n) * 100).toFixed(d) + "%");
const int = (n) => (n === null || n === undefined ? "—" : Number(n).toLocaleString());
const titleCase = (s) => String(s).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

/* CSV values arrive as strings. Long floats (raw similarity scores) are cut to
   4 dp so columns stay scannable; identifiers and labels pass through. */
const ID_COLS = new Set(["Read_ID", "Nearest_Reference_ID"]);
function cellValue(key, v) {
  if (v === null || v === undefined || v === "") return "—";
  const s = String(v);
  if (!ID_COLS.has(key) && /^-?\d+\.\d{5,}$/.test(s)) return Number(s).toFixed(4);
  if (/decision/i.test(key)) return titleCase(s);
  return s.length > 42 ? s.slice(0, 41) + "…" : s;
}

/* ------------------------------------------------------------- chrome --- */

function Loading({ rows = 3 }) {
  return <div>{Array.from({ length: rows }).map((_, i) =>
    <div key={i} className="skel" style={{ height: 74, marginBottom: 12 }} />)}</div>;
}

function ErrorBox({ error }) {
  return <div className="err"><strong>Could not load data.</strong><br />{error}
    <div style={{ marginTop: 8, color: "var(--ink-muted)", fontSize: 12.5 }}>
      Check the API is running: <code>uvicorn src.api:app --port 8000</code>
    </div></div>;
}

/* A render error anywhere below must not blank the whole page -- the worst
   possible failure mode for a dashboard, because it looks identical to "still
   loading". */
class Boundary extends React.Component {
  constructor(p) { super(p); this.state = { err: null }; }
  static getDerivedStateFromError(err) { return { err }; }
  componentDidUpdate(prev) { if (prev.viewKey !== this.props.viewKey && this.state.err) this.setState({ err: null }); }
  render() {
    if (this.state.err) return <div className="err">
      <strong>This view failed to render.</strong><br />{String(this.state.err)}
      <div style={{ marginTop: 8, color: "var(--ink-muted)", fontSize: 12.5 }}>
        Other pages should still work — use the sidebar to continue.</div></div>;
    return this.props.children;
  }
}

function Tile({ label, value, foot, accent }) {
  return <div className="tile">
    <div className="label">{label}</div>
    <div className="value" style={accent ? { color: cssVar(accent) } : null}><CountUp value={value} /></div>
    {foot && <div className="foot">{foot}</div>}
  </div>;
}

function Card({ title, hint, right, children }) {
  return <div className="card">
    {(title || right) && <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
      <div>{title && <h3>{title}</h3>}{hint && <p className="hint">{hint}</p>}</div>
      {right}
    </div>}
    {children}
  </div>;
}

function Legend({ items }) {
  if (!items || items.length < 2) return null;         // one series needs no legend box
  return <div className="legend">{items.map((it) =>
    <span key={it.label}><i style={{ background: cssVar(it.color) }} />{it.label}</span>)}</div>;
}

const TipRows = ({ title, rows }) => <div>
  <div style={{ fontWeight: 650, marginBottom: 4 }}>{title}</div>
  {rows.map(([k, v]) => <div className="t-row" key={k}>
    <span className="t-key">{k}</span><span className="t-val">{v}</span></div>)}
</div>;

/* -------------------------------------------------------------- charts --- */

/* Charts draw at their container's real pixel width (1 SVG unit = 1 px), so
   text keeps its intended size in a wide card, a half-width card or a phone,
   instead of scaling with the card. */
function useChartBox(fallback = 720) {
  const ref = useRef(null);
  const [w, setW] = useState(fallback);
  const [inView, setInView] = useState(!!window.EdnaUI.REDUCED);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, Math.floor(e.contentRect.width))));
    ro.observe(ref.current);
    let io;
    if (!inView) {
      // charts draw in once, when they meaningfully enter the viewport, then stay still
      io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setInView(true); io.disconnect(); } }, { threshold: 0.25 });
      io.observe(ref.current);
    }
    return () => { ro.disconnect(); io && io.disconnect(); };
  }, []);
  return [ref, w, inView];
}
const chartClass = (inView) => "fx-chart" + (inView ? " in" : "");

/* A statement whose words light up from dim to full ink as it scrolls into
   view -- the long-form reveal used on cred.club. segments: [{text, strong}].
   Fully lit when motion is reduced, and lit from the start if the statement is
   already on screen when the page opens, so nothing is ever left unreadable. */
function RevealText({ segments, className = "callout" }) {
  const ref = useRef(null);
  const words = useMemo(() => segments.flatMap((sg) =>
    sg.text.split(/(\s+)/).filter((w) => w !== "").map((w) => ({ w, strong: sg.strong }))), [segments]);
  const count = words.filter((x) => x.w.trim()).length;
  const [lit, setLit] = useState(count);
  useEffect(() => {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { setLit(count); return; }
    const update = () => {
      if (!ref.current) return;
      const r = ref.current.getBoundingClientRect(), vh = innerHeight;
      // 0 when the block's top reaches 90% of the viewport, 1 when its bottom reaches 55%.
      const p = (vh * 0.9 - r.top) / (r.height + vh * 0.35);
      // If the page cannot scroll any further, light everything: otherwise a
      // statement near the bottom of a short page would stay dim for good.
      const atEnd = scrollY + vh >= document.documentElement.scrollHeight - 2;
      setLit(atEnd ? count : Math.round(Math.min(1, Math.max(0, p)) * count));
    };
    update();
    addEventListener("scroll", update, { passive: true });
    addEventListener("resize", update);
    return () => { removeEventListener("scroll", update); removeEventListener("resize", update); };
  }, [count]);
  let i = 0;
  return <div ref={ref} className={className} aria-label={segments.map((s) => s.text).join("")}>
    {words.map((x, k) => {
      if (!x.w.trim()) return x.w;
      const on = i++ < lit;
      const El = x.strong ? "strong" : "span";
      return <El key={k} className="reveal-word" aria-hidden="true" style={{ color: on ? "var(--ink)" : "var(--ink-dim)" }}>{x.w}</El>;
    })}
  </div>;
}

/* Paired before/after values per panel, each panel on its OWN zoomed y-axis.
   For differences far smaller than the values (e.g. F1 0.8806 vs 0.8809), a
   shared zero-based axis draws them identically. Dots joined by a line (a
   slope chart) rather than bars, because a bar's length on a truncated axis
   reads as a false magnitude. panels: [{title, values:[{label, value, color}]}] */
function PairedSlopes({ panels, decimals = 4, height = 250, minSpan = 0.001 }) {
  const f = (v) => Number(v).toFixed(decimals);
  return <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(240px, 100%), 1fr))", gap: 14 }}>
    {panels.map((p) => <SlopePanel key={p.title} p={p} f={f} height={height} minSpan={minSpan} />)}
  </div>;
}

function SlopePanel({ p, f, height, minSpan }) {
  const { show, hide, node } = useTooltip();
  const [ref, outer, inView] = useChartBox(300);
  const W = outer - 10;                      // the panel's own padding
  const H = height, padL = 62, padR = 62, padT = 16, padB = 34;
  const iw = W - padL - padR, ih = H - padT - padB;
  const vals = p.values.map((v) => v.value);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  // Pad around the pair so both points sit inside; never narrower than minSpan,
  // so identical values still get a readable axis.
  const span = Math.max(hi - lo, minSpan);
  const mid = (hi + lo) / 2;
  const min = mid - span * 0.9, max = mid + span * 0.9;
  const y = (v) => padT + ih - ((v - min) / (max - min)) * ih;
  // Points sit inside the plot, clear of the axis labels, so the first value
  // label (drawn to the left of its point) never runs into a tick label.
  const inset = 64;
  const x = (i) => padL + inset + (p.values.length === 1 ? (iw - inset) / 2 : (i / (p.values.length - 1)) * (iw - inset));
  const ticks = 4;
  const delta = vals.length === 2 ? vals[1] - vals[0] : null;
  return <div ref={ref} className={chartClass(inView)} style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "12px 4px 4px" }}>
    <div style={{ display: "flex", justifyContent: "space-between", padding: "0 10px", fontSize: 13 }}>
      <strong>{p.title}</strong>
      {delta !== null && <span className="mono" style={{ color: "var(--ink-2)" }}>Δ {delta > 0 ? "+" : ""}{f(delta)}</span>}
    </div>
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }} role="img"
      aria-label={`${p.title}: ${p.values.map((v) => `${v.label} ${f(v.value)}`).join(", ")}`}>
      {Array.from({ length: ticks + 1 }).map((_, i) => {
        const v = min + ((max - min) * i) / ticks;
        return <g key={i}>
          <line x1={padL} x2={W - 8} y1={y(v)} y2={y(v)} stroke="var(--grid)" strokeWidth="1" />
          <text x={padL - 8} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--ink-muted)" fontFamily="var(--mono)">{f(v)}</text>
        </g>;
      })}
      {p.values.length === 2 && <line className="fx-line" pathLength="1" x1={x(0)} y1={y(vals[0])} x2={x(1)} y2={y(vals[1])} stroke="var(--axis)" strokeWidth="2" />}
      {p.values.map((v, i) => <g key={v.label}>
        <circle className="fx-dot" style={{ "--i": i, cursor: "pointer" }} cx={x(i)} cy={y(v.value)} r="7" fill={cssVar(v.color)} stroke="var(--surface)" strokeWidth="2"
          onMouseMove={(e) => show(e, <TipRows title={`${p.title} · ${v.label}`} rows={[["value", f(v.value)]]} />)}
          onMouseLeave={hide} />
        <text x={x(i) + (i === 0 ? -12 : 12)} y={y(v.value) + 4} textAnchor={i === 0 ? "end" : "start"}
          fontSize="12" fontWeight="600" fill="var(--ink)" fontFamily="var(--mono)"
          style={{ paintOrder: "stroke", stroke: "var(--surface)", strokeWidth: 3 }}>{f(v.value)}</text>
        <text x={x(i)} y={H - 10} textAnchor="middle" fontSize="12" fill="var(--ink-2)">{v.label}</text>
      </g>)}
    </svg>
    {node}
  </div>;
}

/* Grouped bars. Data: [{group, values:[{label,value,color}]}] */
function GroupedBars({ data, height = 260, domain, valueFmt = (v) => fmt(v, 4), axisLabel }) {
  const { show, hide, node } = useTooltip();
  const [ref, W, inView] = useChartBox();
  const padL = 54, padR = 12, padT = 12, padB = 42;
  const H = height;
  const iw = W - padL - padR, ih = H - padT - padB;
  const max = domain ? domain[1] : Math.max(...data.flatMap((g) => g.values.map((v) => v.value))) * 1.12;
  const min = domain ? domain[0] : 0;
  const y = (v) => padT + ih - ((v - min) / (max - min)) * ih;
  const gw = iw / data.length;
  const ticks = 5;

  return <div ref={ref} className={chartClass(inView)}>
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }} role="img">
      {Array.from({ length: ticks + 1 }).map((_, i) => {
        const v = min + ((max - min) * i) / ticks;
        return <g key={i}>
          <line x1={padL} x2={W - padR} y1={y(v)} y2={y(v)} stroke="var(--grid)" strokeWidth="1" />
          <text x={padL - 9} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--ink-muted)">{valueFmt(v)}</text>
        </g>;
      })}
      {axisLabel && <text x={12} y={padT + ih / 2} fontSize="11" fill="var(--ink-muted)"
        transform={`rotate(-90 12 ${padT + ih / 2})`} textAnchor="middle">{axisLabel}</text>}

      {data.map((g, gi) => {
        const n = g.values.length;
        const bw = Math.min(46, (gw - 26) / n);
        const x0 = padL + gi * gw + (gw - bw * n - (n - 1) * 2) / 2;   // 2px surface gap
        return <g key={g.group}>
          {g.values.map((v, vi) => {
            const bx = x0 + vi * (bw + 2), by = y(v.value), bh = Math.max(2, y(min) - by);
            return <rect key={v.label} className="fx-bar" style={{ "--i": gi * n + vi, cursor: "pointer" }} x={bx} y={by} width={bw} height={bh} rx="4" ry="4"
              fill={cssVar(v.color)}
              onMouseMove={(e) => show(e, <TipRows title={`${g.group} · ${v.label}`} rows={[["value", valueFmt(v.value)]]} />)}
              onMouseLeave={hide} />;
          })}
          <text x={padL + gi * gw + gw / 2} y={H - padB + 18} textAnchor="middle"
            fontSize="12" fill="var(--ink-2)">{g.group}</text>
        </g>;
      })}
      <line x1={padL} x2={W - padR} y1={y(min)} y2={y(min)} stroke="var(--axis)" strokeWidth="1" />
    </svg>
    {node}
  </div>;
}

/* Horizontal bars -- best when category labels are long. */
function HBars({ items, valueFmt = (v) => fmt(v, 3), height }) {
  const { show, hide, node } = useTooltip();
  const [ref, W, inView] = useChartBox();
  const rowH = 30, padL = Math.min(224, Math.round(W * 0.42)), padR = 60, padT = 6;
  const H = height || items.length * rowH + padT * 2;
  const iw = W - padL - padR;
  const max = Math.max(...items.map((i) => i.value)) * 1.05 || 1;
  // Labels are right-anchored at padL, so they must fit the gutter or they
  // overflow the viewBox and get clipped at the left edge.
  const MAXC = Math.floor((padL - 14) / 7.2);
  const clip = (s) => (s.length > MAXC ? s.slice(0, MAXC - 1) + "…" : s);

  return <div ref={ref} className={chartClass(inView)}>
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }} role="img">
      {items.map((it, i) => {
        const yy = padT + i * rowH, bh = 15, w = Math.max(2, (it.value / max) * iw);
        return <g key={it.label}>
          <text x={padL - 10} y={yy + bh - 2} textAnchor="end" fontSize="12.5" fill="var(--ink-2)">
            {clip(it.label)}</text>
          <rect className="fx-hbar" style={{ "--i": i, cursor: "pointer" }} x={padL} y={yy} width={w} height={bh} rx="4" ry="4" fill={cssVar(it.color || "--s1")}
            onMouseMove={(e) => show(e, <TipRows title={it.label} rows={[["value", valueFmt(it.value)], ...(it.extra || [])]} />)}
            onMouseLeave={hide} />
          <text x={padL + w + 8} y={yy + bh - 2} fontSize="12" fill="var(--ink-muted)"
            fontVariant="tabular-nums">{valueFmt(it.value)}</text>
        </g>;
      })}
    </svg>
    {node}
  </div>;
}

/* Multi-series line chart with crosshair. series:[{label,color,points:[{x,y}]}] */
function LineChart({ series, height = 250, xLabels, yFmt = (v) => fmt(v, 3), yLabel, domain }) {
  const { show, hide, node } = useTooltip();
  const [ref, W, inView] = useChartBox();
  const padL = 56, padR = 18, padT = 14, padB = 40;
  const H = height, iw = W - padL - padR, ih = H - padT - padB;
  const all = series.flatMap((s) => s.points.map((p) => p.y));
  // Lines default to the data's own range with headroom. Forcing a zero
  // baseline here would flatten every series against the top edge and hide the
  // very differences the chart exists to show; callers that need zero (loss
  // curves, where zero is meaningful) pass an explicit domain.
  const lo = Math.min(...all), hi = Math.max(...all), span = (hi - lo) || Math.abs(hi) || 1;
  const min = domain ? domain[0] : lo - span * 0.35;
  const max = domain ? domain[1] : hi + span * 0.35;
  const xs = series[0]?.points.map((p) => p.x) || [];
  const x = (i) => padL + (xs.length === 1 ? iw / 2 : (i / (xs.length - 1)) * iw);
  const y = (v) => padT + ih - ((v - min) / (max - min || 1)) * ih;
  const ticks = 4;

  return <div ref={ref} className={chartClass(inView)}>
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }} role="img">
      {Array.from({ length: ticks + 1 }).map((_, i) => {
        const v = min + ((max - min) * i) / ticks;
        return <g key={i}>
          <line x1={padL} x2={W - padR} y1={y(v)} y2={y(v)} stroke="var(--grid)" strokeWidth="1" />
          <text x={padL - 9} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--ink-muted)">{yFmt(v)}</text>
        </g>;
      })}
      {yLabel && <text x={12} y={padT + ih / 2} fontSize="11" fill="var(--ink-muted)"
        transform={`rotate(-90 12 ${padT + ih / 2})`} textAnchor="middle">{yLabel}</text>}
      {/* Edge labels anchor inward so the first and last never clip the viewBox. */}
      {xs.map((_, i) => <text key={i} x={x(i)} y={H - padB + 20} fontSize="12" fill="var(--ink-2)"
        textAnchor={i === 0 ? "start" : i === xs.length - 1 ? "end" : "middle"}>
        {xLabels ? xLabels[i] : xs[i]}</text>)}

      {series.map((s) => <g key={s.label}>
        <path className="fx-line" pathLength="1" d={s.points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.y)}`).join(" ")}
          fill="none" stroke={cssVar(s.color)} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        {s.points.map((p, i) => <circle key={i} className="fx-dot" style={{ "--i": i, cursor: "pointer" }} cx={x(i)} cy={y(p.y)} r="4.5"
          fill={cssVar(s.color)} stroke="var(--surface)" strokeWidth="2"
          onMouseMove={(e) => show(e, <TipRows title={s.label} rows={[[xLabels ? xLabels[i] : `x ${p.x}`, yFmt(p.y)]]} />)}
          onMouseLeave={hide} />)}
      </g>)}
      <line x1={padL} x2={W - padR} y1={y(min)} y2={y(min)} stroke="var(--axis)" strokeWidth="1" />
    </svg>
    {node}
  </div>;
}

/* Stacked single bar showing decision-tier composition. */
function StackedTiers({ tiers, total }) {
  const { show, hide, node } = useTooltip();
  const order = ["assigned", "best_guess", "best_guess_low_confidence", "unclassified"];
  const colorOf = (k) => ({ assigned: "--s3", best_guess: "--s1",
    best_guess_low_confidence: "--s4", unclassified: "--s8" }[k] || "--s7");
  const keys = order.filter((k) => tiers[k]).concat(Object.keys(tiers).filter((k) => !order.includes(k)));
  const [ref, W, inView] = useChartBox();
  let acc = 0;
  return <div ref={ref} className={chartClass(inView)}>
    <svg width={W} height="34" viewBox={`0 0 ${W} 34`} style={{ display: "block" }} role="img">
      {keys.map((k) => {
        const w = (tiers[k] / total) * W, xx = acc; acc += w;
        return <rect key={k} className="fx-hbar" style={{ "--i": keys.indexOf(k), cursor: "pointer" }} x={xx + (xx ? 1 : 0)} y="4" width={Math.max(0, w - 2)} height="22" rx="4" ry="4"
          fill={cssVar(colorOf(k))}
          onMouseMove={(e) => show(e, <TipRows title={titleCase(k)}
            rows={[["reads", int(tiers[k])], ["share", ((tiers[k] / total) * 100).toFixed(1) + "%"]]} />)}
          onMouseLeave={hide} />;
      })}
    </svg>
    <div className="legend" style={{ marginTop: 10 }}>
      {keys.map((k) => <span key={k}><i style={{ background: cssVar(colorOf(k)) }} />
        {titleCase(k)} · {int(tiers[k])}</span>)}
    </div>
    {node}
  </div>;
}

/* ---------------------------------------------------------------- views --- */

/* Test-set sizes present in the saved PRF results, ascending. */
const prfSizes = (all) => [...new Set(all.kingdoms.flatMap((k) => k.results.map((r) => r.n)))].sort((a, b) => a - b);
const signed = (n) => (n > 0 ? "+" : "") + n;
const kName = (k) => titleCase(k).replace(" Large", " (large)");
/* Embedding-only and BM25-only arms in an architecture evaluation, found by
   their naming scheme (A_* = BM25, C_*_frozen / C_*_finetuned = DNABERT alone). */
const armOf = (arms, re) => arms.find((a) => re.test(a.architecture));

function Overview({ go }) {
  const all = useApi("/metrics/prf");
  const arch = useApi("/metrics/architecture");
  const nMax = all.data ? prfSizes(all.data).slice(-1)[0] : null;
  const prf = useApi(nMax ? `/metrics/prf/comparison?n=${nMax}` : null, [nMax]);
  if (all.loading || prf.loading || arch.loading || (nMax && !prf.data && !prf.error)) return <Loading rows={4} />;
  if (all.error || prf.error) return <ErrorBox error={all.error || prf.error} />;

  const ks = prf.data.kingdoms;
  const totalFrags = ks.reduce((a, k) => a + k.n, 0);
  const netCalls = ks.reduce((a, k) => a + k.delta.correct_calls, 0);
  const best = ks.flatMap((k) => ["frozen", "finetuned"].map((arm) => ({ k, arm, f1: k[arm].f1 })))
    .reduce((a, b) => (b.f1 > a.f1 ? b : a));
  const indep = ks.filter((k) => k.fully_independent_samples).map((k) => kName(k.kingdom));
  const resampled = ks.filter((k) => !k.fully_independent_samples).map((k) => kName(k.kingdom));

  // Isolated effect of fine-tuning (embedding-only retrieval) and what BM25 finds alone.
  let iso = null, bm25 = [];
  if (arch.data) {
    for (const kd of arch.data.kingdoms) {
      const fr = armOf(kd.architectures, /^C_.*frozen$/), ft = armOf(kd.architectures, /^C_.*finetuned$/);
      const a = armOf(kd.architectures, /^A_/);
      if (fr && ft) {
        const gain = ft.recall_at_10 - fr.recall_at_10;
        if (!iso || gain > iso.gain) iso = { gain, kingdom: kd.kingdom, fr: fr.recall_at_10, ft: ft.recall_at_10 };
      }
      if (a) bm25.push(a.recall_at_10);
    }
  }

  return <div>
    <div className="grid k4" style={{ marginBottom: 18 }}>
      <Tile label="Kingdoms evaluated" value={ks.length} foot={ks.map((k) => kName(k.kingdom)).join(" · ")} />
      <Tile label="Held-out fragments" value={int(totalFrags)} foot={`${ks.length} kingdoms × n = ${int(nMax)}`} />
      <Tile label="Net effect of fine-tuning" value={signed(netCalls)}
        foot={`Correct Calls, Of ${int(totalFrags)}`} />
      <Tile label="Best F1" value={fmt(best.f1, 4)} foot={`${kName(best.k.kingdom)} · ${best.arm === "frozen" ? "Frozen" : "Fine-Tuned"}`} accent="--s3" />
    </div>

    <Card title="Frozen vs fine-tuned F1, per kingdom"
      hint={`${int(nMax)} held-out fragments each. Fine-tuning changed ${signed(netCalls)} correct calls of ${int(totalFrags)} (${(100 * netCalls / totalFrags).toFixed(3)}%).`}>
      <RevealText segments={[
        ...(iso ? [
          { text: "Measured in isolation, fine-tuning raised embedding-only retrieval (recall@10) by up to " },
          { text: `${signed((iso.gain * 100).toFixed(1))} percentage points`, strong: true },
          { text: ` (${titleCase(iso.kingdom)}: ${fmt(iso.fr, 3)} → ${fmt(iso.ft, 3)}). ` }] : []),
        { text: "Inside the full pipeline it is worth " },
        { text: `${signed(netCalls)} correct calls out of ${int(totalFrags)}`, strong: true },
        { text: "." },
        ...(bm25.length ? [{ text: ` BM25 alone already reaches recall@10 of ${fmt(Math.min(...bm25), 3)}${bm25.length > 1 ? `–${fmt(Math.max(...bm25), 3)}` : ""} across kingdoms, and the embedding model only re-orders candidates BM25 has already found.` }] : []),
      ]} />
      <Legend items={[{ label: "Frozen", color: "--s1" }, { label: "Fine-tuned", color: "--s2" }]} />
      <PairedSlopes decimals={4} panels={ks.map((k) => ({ title: kName(k.kingdom), values: [
        { label: "Frozen", value: k.frozen.f1, color: "--s1" },
        { label: "Fine-tuned", value: k.finetuned.f1, color: "--s2" }] }))} />
      <p className="hint" style={{ marginTop: 12, marginBottom: 0 }}>
        Each kingdom has its own y-axis, zoomed to its pair of values, so differences in the 4th decimal are visible.
        Compare within a panel, not across panels.
      </p>
    </Card>

    <Card title="Per-kingdom detail" hint={`At n = ${int(nMax)} held-out fragments.`}>
      <div className="scroll-x"><table>
        <thead><tr><th>Kingdom</th><th>Precision</th><th>Recall</th><th>F1 frozen</th><th>F1 fine-tuned</th>
          <th>Δ calls</th><th>Samples</th></tr></thead>
        <tbody>{ks.map((k) => <tr key={k.kingdom}>
          <td>{kName(k.kingdom)}</td>
          <td>{fmt(k.finetuned.precision)}</td>
          <td>{fmt(k.finetuned.recall)}</td>
          <td>{fmt(k.frozen.f1)}</td>
          <td>{fmt(k.finetuned.f1)}</td>
          <td style={{ color: k.delta.correct_calls > 0 ? "var(--good-ink)" : k.delta.correct_calls < 0 ? "var(--critical)" : "var(--ink-muted)" }}>
            {signed(k.delta.correct_calls)}</td>
          <td>{k.fully_independent_samples
            ? <span className="pill good">independent</span>
            : <span className="pill warn">{k.fragments_per_reference.toFixed(2)}/ref</span>}</td>
        </tr>)}</tbody>
      </table></div>
      <p className="hint" style={{ marginTop: 14, marginBottom: 0 }}>
        “Samples” flags whether fragments are statistically independent (one fragment per held-out reference).
        {indep.length > 0 && <> Independent: {indep.join(", ")}.</>}
        {resampled.length > 0 && <> Re-sampled, so their fragments are correlated: {resampled.join(", ")}.</>}
      </p>
    </Card>

    <div className="grid k2" style={{ marginTop: 18 }}>
      <Card title="Explore the metrics" hint={`Precision, recall and F1 at every test-set size (${prfSizes(all.data).map(int).join(", ")}).`}>
        <button className="btn" onClick={() => go("metrics")}>Open metrics →</button></Card>
      <Card title="Explore real reads" hint="Per-read classifications from real seawater samples.">
        <button className="btn" onClick={() => go("reads")}>Open read explorer →</button></Card>
    </div>
  </div>;
}

function Metrics() {
  const all = useApi("/metrics/prf");
  const sizes = all.data ? prfSizes(all.data) : [];
  const [picked, setN] = useState(null);
  const n = picked || sizes[sizes.length - 1] || null;
  const cmp = useApi(n ? `/metrics/prf/comparison?n=${n}` : null, [n]);
  if (all.loading || cmp.loading || (n && !cmp.data && !cmp.error)) return <Loading rows={4} />;
  if (all.error || cmp.error) return <ErrorBox error={all.error || cmp.error} />;

  const ks = cmp.data.kingdoms;
  // Axis floor from the data: the lowest score shown, rounded down to the next 0.05.
  const scores = ks.flatMap((k) => [k.finetuned.precision, k.finetuned.recall, k.finetuned.f1]);
  const floor = Math.max(0, Math.floor((Math.min(...scores) - 0.02) * 20) / 20);
  return <div>
    <div className="filters">
      <label>Test-set size</label>
      {sizes.map((v) =>
        <button key={v} className="btn" onClick={() => setN(v)}
          style={n === v ? { background: "var(--surface-2)", fontWeight: 650, borderColor: "var(--s1)" } : null}>
          {int(v)}</button>)}
    </div>

    <Card title="Precision, recall and F1" hint={`Production pipeline scored as a decision system, n = ${int(n)} fragments.`}>
      <Legend items={[{ label: "Precision", color: "--s1" }, { label: "Recall", color: "--s2" }, { label: "F1", color: "--s3" }]} />
      <GroupedBars domain={[floor, 1.0]} valueFmt={(v) => v.toFixed(2)} axisLabel="Score"
        data={ks.map((k) => ({ group: titleCase(k.kingdom).replace(" Large", " (lg)"),
          values: [{ label: "Precision", value: k.finetuned.precision, color: "--s1" },
                   { label: "Recall", value: k.finetuned.recall, color: "--s2" },
                   { label: "F1", value: k.finetuned.f1, color: "--s3" }] }))} />
    </Card>

    <Card title="Frozen vs fine-tuned" hint="Same fragments, same pipeline — only the embedding model differs.">
      <div className="scroll-x"><table>
        <thead><tr><th>Kingdom</th><th>Arm</th><th>Precision</th><th>Recall</th><th>F1</th>
          <th>TP</th><th>Δ F1</th><th>Δ calls</th></tr></thead>
        <tbody>{ks.flatMap((k) => [
          <tr key={k.kingdom + "f"}><td rowSpan="2">{titleCase(k.kingdom)}</td><td>Frozen</td>
            <td>{fmt(k.frozen.precision)}</td><td>{fmt(k.frozen.recall)}</td><td>{fmt(k.frozen.f1)}</td>
            <td>{int(k.frozen.tp)}</td><td rowSpan="2">{k.delta.f1 >= 0 ? "+" : ""}{fmt(k.delta.f1, 4)}</td>
            <td rowSpan="2" style={{ color: k.delta.correct_calls > 0 ? "var(--good-ink)" : k.delta.correct_calls < 0 ? "var(--critical)" : "var(--ink-muted)" }}>
              {k.delta.correct_calls > 0 ? "+" : ""}{k.delta.correct_calls}</td></tr>,
          <tr key={k.kingdom + "t"}><td>Fine-Tuned</td>
            <td>{fmt(k.finetuned.precision)}</td><td>{fmt(k.finetuned.recall)}</td><td>{fmt(k.finetuned.f1)}</td>
            <td>{int(k.finetuned.tp)}</td></tr>])}</tbody>
      </table></div>
    </Card>

    <Card title="Stability across test-set size" hint={`Nested prefixes of one ${int(sizes[sizes.length - 1])}-fragment run, so sizes are directly comparable.`}>
      <Legend items={all.data.kingdoms.map((k, i) => ({ label: titleCase(k.kingdom), color: SERIES[i] }))} />
      <LineChart yLabel="F1 (fine-tuned)" yFmt={(v) => v.toFixed(3)}
        xLabels={sizes.map(int)}
        series={all.data.kingdoms.map((k, i) => ({
          label: titleCase(k.kingdom), color: SERIES[i],
          points: sizes.map((sz, j) => ({ x: j,
            y: (k.results.find((r) => r.n === sz && r.arm === "finetuned") || {}).f1 || 0 })) }))} />
      <p className="hint" style={{ marginTop: 12, marginBottom: 0 }}>
        Flat lines mean the larger samples confirmed the result rather than revealing a hidden effect.
      </p>
    </Card>

    <Card title="Split composition" hint="What each kingdom's held-out set is made of.">
      <div className="scroll-x"><table>
        <thead><tr><th>Kingdom</th><th>Train refs</th><th>Held-out refs</th><th>Fragments</th>
          <th>Frags / ref</th><th>Independent</th></tr></thead>
        <tbody>{all.data.kingdoms.map((k) => <tr key={k.kingdom}>
          <td>{titleCase(k.kingdom)}</td><td>{int(k.split.n_train_refs)}</td>
          <td>{int(k.split.n_holdout_refs)}</td><td>{int(k.split.n_fragments)}</td>
          <td>{k.split.fragments_per_reference.toFixed(2)}</td>
          <td>{k.split.fully_independent_samples
            ? <span className="pill good">yes</span> : <span className="pill warn">re-sampled</span>}</td>
        </tr>)}</tbody></table></div>
    </Card>
  </div>;
}

function Architecture() {
  const arch = useApi("/metrics/architecture");
  const [picked, setKingdom] = useState(null);
  if (arch.loading) return <Loading rows={3} />;
  if (arch.error) return <ErrorBox error={arch.error} />;
  const available = arch.data.kingdoms.map((k) => k.kingdom);
  const kingdom = picked || available[0];
  const kd = arch.data.kingdoms.find((k) => k.kingdom === kingdom);
  const arms = kd.architectures;
  // Candidate redundancy, frozen vs fine-tuned, for every arm that has both.
  const pairs = arms.filter((a) => /_finetuned$/.test(a.architecture)).map((ft) => {
    const fr = arms.find((a) => a.architecture === ft.architecture.replace(/_finetuned$/, "_frozen"));
    return fr && { name: ft.architecture.replace(/_finetuned$/, ""), fr: fr.mean_pairwise_sim, ft: ft.mean_pairwise_sim };
  }).filter(Boolean);
  const lowered = pairs.filter((p) => p.ft < p.fr);
  const label = (a) => a.replace(/_/g, " ").replace("BM25 MMR", "BM25→MMR").replace("RRF", "RRF");

  return <div>
    <div className="filters">
      <label>Kingdom</label>
      {available.map((k) =>
        <button key={k} className="btn" onClick={() => setKingdom(k)}
          style={kingdom === k ? { background: "var(--surface-2)", fontWeight: 650, borderColor: "var(--s1)" } : null}>
          {titleCase(k)}</button>)}
    </div>

    <div className="grid k3" style={{ marginBottom: 18 }}>
      <Tile label="Held-out fragments" value={int(kd.n_holdout_fragments)} />
      <Tile label="Identity threshold" value={kd.identity_threshold + "%"} foot="Bar For A Confident Call" />
      <Tile label="Architectures compared" value={arms.length} foot="Each Through The Same Reranker" />
    </div>

    <Card title="Recall@10 by architecture" hint="Whether the true match appears in the final top-10 candidate set.">
      <HBars valueFmt={(v) => v.toFixed(3)}
        items={arms.map((a) => ({ label: label(a.architecture), value: a.recall_at_10,
          color: a.architecture.startsWith("A") ? "--s3" : a.architecture.includes("finetuned") ? "--s2" : "--s1",
          extra: [["recall@1", fmt(a.recall_at_1, 3)], ["assigned", int(a.n_assigned)]] }))} />
      <Legend items={[{ label: "BM25 alone", color: "--s3" }, { label: "Frozen embeddings", color: "--s1" },
                      { label: "Fine-tuned embeddings", color: "--s2" }]} />
    </Card>

    <Card title="Full comparison" hint="Recall, assignment rate, accuracy and candidate redundancy for every arm.">
      <div className="scroll-x"><table>
        <thead><tr><th>Architecture</th><th>R@1</th><th>R@5</th><th>R@10</th>
          <th>Assigned</th><th>Accuracy</th><th>Redundancy</th></tr></thead>
        <tbody>{arms.map((a) => <tr key={a.architecture}>
          <td>{label(a.architecture)}</td>
          <td>{fmt(a.recall_at_1, 3)}</td><td>{fmt(a.recall_at_5, 3)}</td><td>{fmt(a.recall_at_10, 3)}</td>
          <td>{int(a.n_assigned)}</td>
          <td>{a.accuracy_on_assigned === null ? "—" : fmt(a.accuracy_on_assigned, 3)}</td>
          <td>{fmt(a.mean_pairwise_sim, 3)}</td></tr>)}</tbody>
      </table></div>
      {pairs.length > 0 && <p className="hint" style={{ marginTop: 14, marginBottom: 0 }}>
        Lower redundancy means a more diverse candidate set. Here, fine-tuned embeddings lowered it in{" "}
        {lowered.length} of {pairs.length} arm{pairs.length === 1 ? "" : "s"}
        {pairs.length > 0 && <> ({pairs.map((p) => `${label(p.name)}: ${fmt(p.fr, 3)} → ${fmt(p.ft, 3)}`).join("; ")})</>}.
      </p>}
    </Card>
  </div>;
}

function Models() {
  const models = useApi("/models");
  const cmp = useApi("/models/comparison");
  if (models.loading || cmp.loading) return <Loading rows={4} />;
  if (models.error) return <ErrorBox error={models.error} />;

  const corrected = models.data.models.filter((m) => m.objective && m.mean_loss_per_epoch);
  const cmpRows = Object.entries(cmp.data.models)
    .map(([k, v]) => ({ label: k === "dnaberts_frozen_baseline" ? "DNABERT-S (production)" : titleCase(k),
      value: v.recall_at_10 ?? 0, r1: v.recall_at_1 }))
    .sort((a, b) => b.value - a.value);

  return <div>
    <Card title="Embedding model comparison"
      hint={`Recall@10 — ${cmp.data.test_set}. Higher is better.`}>
      <HBars valueFmt={(v) => (v * 100).toFixed(1) + "%"}
        items={cmpRows.map((r, i) => ({ label: r.label, value: r.value,
          color: i === 0 ? "--s3" : "--s1", extra: [["recall@1", fmt(r.r1, 3)]] }))} />
      {cmpRows.length > 1 && <p className="hint" style={{ marginTop: 10, marginBottom: 0 }}>
        {cmpRows[0].label} leads with recall@10 of {(cmpRows[0].value * 100).toFixed(1)}%
        {cmpRows[1].value > 0 && <>, {(cmpRows[0].value / cmpRows[1].value).toFixed(1)}× the next best
          ({cmpRows[1].label}, {(cmpRows[1].value * 100).toFixed(1)}%)</>}.
      </p>}
    </Card>

    <Card title="Fine-tuning loss curves" hint="Corrected (v2) training runs. Lower is better, but see the note.">
      <Legend items={corrected.map((m, i) => ({ label: titleCase(m.kingdom), color: SERIES[i % SERIES.length] }))} />
      <LineChart yLabel="Mean Loss" yFmt={(v) => v.toFixed(2)}
        xLabels={Array.from({ length: Math.max(0, ...corrected.map((m) => m.mean_loss_per_epoch.length)) }, (_, i) => `Epoch ${i + 1}`)}
        domain={[0, Math.max(...corrected.flatMap((m) => m.mean_loss_per_epoch)) * 1.1]}
        series={corrected.map((m, i) => ({ label: titleCase(m.kingdom), color: SERIES[i % SERIES.length],
          points: m.mean_loss_per_epoch.map((v, j) => ({ x: j, y: v })) }))} />
      <div className="callout" style={{ marginTop: 14 }}>
        A falling training loss is <strong>not</strong> evidence of better retrieval — the held-out metrics
        (Overview, Precision / Recall, Architecture) are what settle it.
      </div>
    </Card>

    <Card title="All checkpoints" hint="Every fine-tuning run recorded in this project.">
      <div className="scroll-x"><table>
        <thead><tr><th>Checkpoint</th><th>Kingdom</th><th>Train refs</th><th>Distinct seqs</th>
          <th>Epochs</th><th>Loss 1 → 2</th><th>Minutes</th></tr></thead>
        <tbody>{models.data.models.map((m) => <tr key={m.name}>
          <td className="mono">{m.name.replace("dnaberts_finetuned_", "")}</td>
          <td>{m.kingdom ? titleCase(m.kingdom) : "—"}</td>
          <td>{m.n_train_references ? int(m.n_train_references) : "—"}</td>
          <td>{m.n_distinct_sequences ? int(m.n_distinct_sequences) : "—"}</td>
          <td>{m.epochs ?? "—"}</td>
          <td>{m.mean_loss_per_epoch ? m.mean_loss_per_epoch.map((v) => v.toFixed(3)).join(" → ") : "—"}</td>
          <td>{m.train_seconds ? (m.train_seconds / 60).toFixed(0) : "—"}</td></tr>)}</tbody>
      </table></div>
      <p className="hint" style={{ marginTop: 14, marginBottom: 0 }}>
        A dash means that checkpoint's metadata does not record the field.
      </p>
    </Card>
  </div>;
}

function Reads() {
  const list = useApi("/reads/datasets");
  const [ds, setDs] = useState(null);
  const [decision, setDecision] = useState("");
  const [minScore, setMinScore] = useState("");
  const [page, setPage] = useState(0);
  const LIMIT = 25;

  const active = ds || (list.data && list.data.datasets.find((d) => d.n_reads) || {}).dataset;
  const q = useMemo(() => {
    const p = new URLSearchParams({ limit: LIMIT, offset: page * LIMIT });
    if (decision) p.set("decision", decision);
    if (minScore !== "") p.set("min_score", minScore);
    return p.toString();
  }, [decision, minScore, page]);

  const rows = useApi(active ? `/reads/${active}?${q}` : null, [active, q]);
  const summary = useApi(active ? `/reads/${active}/summary?top=8` : null, [active]);

  if (list.loading) return <Loading rows={4} />;
  if (list.error) return <ErrorBox error={list.error} />;
  const meta = list.data.datasets.find((d) => d.dataset === active) || {};

  return <div>
    <div className="filters">
      <label>Dataset</label>
      <select value={active || ""} onChange={(e) => { setDs(e.target.value); setPage(0); }}>
        {list.data.datasets.filter((d) => d.n_reads).map((d) =>
          <option key={d.dataset} value={d.dataset}>{titleCase(d.dataset)} · {int(d.n_reads)} Reads</option>)}
      </select>
      <label>Decision</label>
      <select value={decision} onChange={(e) => { setDecision(e.target.value); setPage(0); }}>
        <option value="">All</option>
        {Object.keys((summary.data && summary.data.decisions) || {}).map((k) =>
          <option key={k} value={k}>{titleCase(k)}</option>)}
      </select>
      <label>Min score</label>
      <input type="number" step="1" placeholder="any" style={{ width: 92 }}
        value={minScore} onChange={(e) => { setMinScore(e.target.value); setPage(0); }} />
      {(decision || minScore !== "") &&
        <button className="btn" onClick={() => { setDecision(""); setMinScore(""); setPage(0); }}>Clear</button>}
    </div>

    {meta.description && <div className="callout" style={{ marginBottom: 18 }}>
      <strong>{meta.part} · {meta.accession}</strong> — {meta.description}</div>}

    {summary.data && summary.data.decisions && <>
      <div className="grid k4" style={{ marginBottom: 18 }}>
        <Tile label="Reads" value={int(summary.data.n_reads)} foot={titleCase(summary.data.kingdom)} />
        <Tile label="Confidently assigned" value={int(summary.data.decisions.assigned || 0)}
          accent="--s3"
          foot={pct((summary.data.decisions.assigned || 0) / summary.data.n_reads)} />
        <Tile label="Median score" value={fmt(summary.data.score_distribution.median, 1)}
          foot={<span className="mono">{summary.data.score_column}</span>} />
        <Tile label="Distinct organisms" value={int(summary.data.distinct_values)}
          foot={<>Grouped By <span className="mono">{summary.data.grouped_by}</span></>} />
      </div>

      <Card title="Confidence composition" hint="How the pipeline's decisions break down across this dataset.">
        <StackedTiers tiers={summary.data.decisions} total={summary.data.n_reads} />
      </Card>

      <div className="grid k2" style={{ marginTop: 18 }}>
        <Card title="Score distribution" hint={`Percentiles of ${summary.data.score_column}.`}>
          <HBars valueFmt={(v) => v.toFixed(2)} items={[
            ["Min", summary.data.score_distribution.min], ["P25", summary.data.score_distribution.p25],
            ["Median", summary.data.score_distribution.median], ["P75", summary.data.score_distribution.p75],
            ["Max", summary.data.score_distribution.max]]
            .map(([k, v], i) => ({ label: k, value: v ?? 0, color: SERIES[0] }))} />
        </Card>
        <Card title="Top matches" hint={`Most frequent ${summary.data.grouped_by}.`}>
          <HBars valueFmt={(v) => int(v)} items={summary.data.top.map((t) => ({
            label: t.value, value: t.reads, color: "--s1",
            extra: [["share", t.pct_of_total + "%"]] }))} />
        </Card>
      </div>
    </>}

    <Card title="Reads" hint={rows.data ? `${int(rows.data.total_matching)} matching · showing ${page * LIMIT + 1}–${Math.min((page + 1) * LIMIT, rows.data.total_matching)}` : ""}
      right={rows.data && <div style={{ display: "flex", gap: 8 }}>
        <button className="btn" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>← Prev</button>
        <button className="btn" disabled={(page + 1) * LIMIT >= rows.data.total_matching}
          onClick={() => setPage((p) => p + 1)}>Next →</button></div>}>
      {rows.loading || !rows.data ? <Loading rows={3} /> : rows.error ? <ErrorBox error={rows.error} /> :
        <div className="scroll-x"><table>
          <thead><tr>{(rows.data.reads[0] ? Object.keys(rows.data.reads[0]) : []).map((c) =>
            <th key={c} className={c === "Read_ID" ? "mono" : ""}>{c.replace(/_/g, " ")}</th>)}</tr></thead>
          <tbody>{rows.data.reads.map((r, i) => <tr key={i}>
            {Object.entries(r).map(([k, v]) => <td key={k}
              className={k === "Read_ID" || k === "Nearest_Reference_ID" ? "mono" : ""}
              style={k === "Confidence_Decision" ? {
                color: v === "assigned" ? "var(--good-ink)" : v === "best_guess" ? "var(--ink)" : "var(--ink-muted)",
                fontWeight: v === "assigned" ? 600 : 400 } : null}>
              {cellValue(k, v)}</td>)}
          </tr>)}</tbody>
        </table>
        {rows.data.reads.length === 0 && <p className="hint" style={{ marginTop: 14 }}>No reads match these filters.</p>}
        </div>}
    </Card>
  </div>;
}

function Datasets() {
  const list = useApi("/reads/datasets");
  const kingdoms = useApi("/kingdoms");
  if (list.loading || kingdoms.loading) return <Loading rows={4} />;
  if (list.error) return <ErrorBox error={list.error} />;
  return <div>
    <Card title="Per-read result sets" hint="Every saved classification run in this project. Counts are read from each file.">
      <div className="scroll-x"><table>
        <thead><tr><th>Dataset</th><th>Kingdom</th><th>Part</th><th>Accession</th><th>Reads</th><th>Assigned</th><th>Score column</th></tr></thead>
        <tbody>{list.data.datasets.map((d) => <tr key={d.dataset}>
          <td><div style={{ fontWeight: 600 }}>{titleCase(d.dataset)}</div>
            <div style={{ fontSize: 12, color: "var(--ink-muted)", maxWidth: 520, whiteSpace: "normal" }}>{d.description}</div></td>
          <td>{titleCase(d.kingdom)}</td><td>{d.part}</td><td className="mono">{d.accession}</td>
          <td>{int(d.n_reads)}</td>
          <td>{d.n_reads ? `${int(d.n_assigned)} (${(100 * d.n_assigned / d.n_reads).toFixed(1)}%)` : "—"}</td>
          <td className="mono">{d.score_column || "—"}</td></tr>)}</tbody>
      </table></div>
    </Card>
    <Card title="Data coverage by kingdom" hint="Which evaluations exist for each kingdom.">
      <div className="scroll-x"><table>
        <thead><tr><th>Kingdom</th><th>Precision/Recall/F1</th><th>Architecture eval</th><th>Read datasets</th></tr></thead>
        <tbody>{Object.entries(kingdoms.data).map(([k, v]) => <tr key={k}>
          <td>{titleCase(k)}</td>
          <td>{v.precision_recall_f1 ? <span className="pill good">yes</span> : <span className="pill">—</span>}</td>
          <td>{v.architecture_evaluation ? <span className="pill good">yes</span> : <span className="pill">—</span>}</td>
          <td>{v.read_datasets.length || "—"}</td></tr>)}</tbody>
      </table></div>
    </Card>
  </div>;
}

/* ----------------------------------------------------------------- app --- */

/* The analysis pages come from web/analysis.jsx (loaded first, as its own
   module) and render inside an .ax wrapper that scopes their styles. */
const AX = window.EdnaAnalysis;
const ax = (C) => (props) => <div className="ax"><C {...props} /></div>;

/* Line icons for the navigation (18px, currentColor). */
const NavIcon = ({ name }) => {
  const paths = {
    analyze: <><path d="M12 15V4M7.5 8.5 12 4l4.5 4.5" /><path d="M4 14v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" /></>,
    runs: <><path d="M4 6h16M4 12h16M4 18h10" /></>,
    overview: <><rect x="4" y="4" width="7" height="7" rx="1.5" /><rect x="13" y="4" width="7" height="7" rx="1.5" /><rect x="4" y="13" width="7" height="7" rx="1.5" /><rect x="13" y="13" width="7" height="7" rx="1.5" /></>,
    metrics: <><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></>,
    architecture: <><circle cx="6" cy="6" r="2.2" /><circle cx="18" cy="6" r="2.2" /><circle cx="12" cy="18" r="2.2" /><path d="M7.6 7.6 10.6 16M16.4 7.6 13.4 16M8.2 6h7.6" /></>,
    models: <><path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" /><path d="M4 7.5 12 12l8-4.5M12 12v9" /></>,
    reads: <><circle cx="11" cy="11" r="6.5" /><path d="m20 20-4.2-4.2" /></>,
    datasets: <><ellipse cx="12" cy="6" rx="7.5" ry="2.8" /><path d="M4.5 6v12c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8V6M4.5 12c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8" /></>,
    theme: <><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" /></>,
    docs: <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5M9 13h6M9 17h6" /></>,
  };
  return <svg className="nav-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
};

const PAGES = {
  analyze:      { section: "Your Samples", title: "Analyse a Sample", sub: "Upload reads and identify the species in them.", C: ax(AX.AnalyzeView) },
  runs:         { title: "Runs",                sub: "Every uploaded file and its analysis.",             C: ax(AX.RunsView) },
  overview:     { section: "Project Results", title: "Overview", sub: "Headline results across all three kingdoms.", C: Overview },
  metrics:      { title: "Precision / Recall",  sub: "The pipeline scored as a decision system.",         C: Metrics },
  architecture: { title: "Architecture",        sub: "Seven retrieval architectures, head to head.",      C: Architecture },
  models:       { title: "Models",              sub: "Fine-tuning runs and embedding-model comparison.",  C: Models },
  reads:        { title: "Read Explorer",       sub: "Per-read classifications from real seawater.",      C: Reads },
  datasets:     { title: "Datasets",            sub: "What data exists and where it came from.",          C: Datasets },
};
// Reached from the Runs page, not the sidebar: "#run/<id>", "#compare/<id,id>".
const SUBPAGES = {
  run:     { parent: "runs", title: "Run",          sub: "Findings for one uploaded file.",  C: ax(AX.RunView) },
  compare: { parent: "runs", title: "Compare Runs", sub: "Uploaded runs side by side.",      C: ax(AX.CompareView) },
};

function useTheme() {
  /* Resolve to an explicit "light"/"dark" on first paint rather than leaving it
     empty. With an empty value the OS preference wins, but the toggle then has
     to be pressed twice before anything visibly changes -- the first press only
     makes the implicit choice explicit. */
  const [theme, setTheme] = useState(() => {
    try {
      const saved = localStorage.getItem("edna-theme");
      if (saved === "light" || saved === "dark") return saved;
    } catch (_) {}
    return "dark";
  });
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("edna-theme", theme); } catch (_) {}
  }, [theme]);
  return [theme, setTheme];
}

/* The active-item indicator glides between navigation items (spring), rather
   than the highlight jumping. Measured from the DOM, so it follows any layout. */
function useNavIndicator(active) {
  const navRef = useRef(null);
  const [box, setBox] = useState(null);
  const measure = useCallback(() => {
    const nav = navRef.current;
    const el = nav && nav.querySelector(".navlink.active");
    if (el) setBox({ y: el.offsetTop, h: el.offsetHeight });
  }, []);
  useLayoutEffect(measure, [active]);
  useEffect(() => { addEventListener("resize", measure); return () => removeEventListener("resize", measure); }, []);
  return [navRef, box];
}

function App() {
  const parse = () => { const [view, arg = null] = (location.hash || "#overview").slice(1).split("/"); return { view, arg }; };
  const [route, setRoute] = useState(parse);
  const [theme, setTheme] = useTheme();
  const [chatOpen, setChatOpen] = useState(false);
  const [chatRun, setChatRun] = useState(null);
  const [menuOpen, setMenuOpen] = useState(false);
  useEffect(() => {
    const on = () => { setRoute(parse()); setMenuOpen(false); scrollTo({ top: 0, behavior: "instant" }); };
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  useEffect(() => { document.body.classList.toggle("chat-open", chatOpen); }, [chatOpen]);
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") { setChatOpen(false); setMenuOpen(false); } };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, []);
  // The first view has painted: the loader can hand over to the logo (web/fx.js).
  useEffect(() => {
    requestAnimationFrame(() => requestAnimationFrame(() => window.EDNAFX && window.EDNAFX.ready && window.EDNAFX.ready()));
  }, []);

  const go = (p) => { location.hash = "#" + p; };
  // The assistant opens about the run on screen, or about the whole project.
  const openChat = (runId) => { setChatRun(runId || null); setChatOpen(true); };
  const cur = SUBPAGES[route.view] || PAGES[route.view] || PAGES.overview;
  const active = SUBPAGES[route.view] ? SUBPAGES[route.view].parent : (PAGES[route.view] ? route.view : "overview");
  const View = cur.C;
  const [navRef, box] = useNavIndicator(active);

  const brand = <a className="brand-mark" href="#overview" aria-label="eDNA-TaxAI home">
    <HelixLogo /><h1 className="keep-case">eDNA-TaxAI</h1></a>;

  return <div className="shell">
    <div className="mobile-bar">
      {brand}
      <button className="menu-btn" aria-label="Open navigation" aria-expanded={menuOpen} onClick={() => setMenuOpen(!menuOpen)}>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
          {menuOpen ? <path d="M6 6l12 12M18 6 6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}</svg></button>
    </div>
    {menuOpen && <div className="scrim" onClick={() => setMenuOpen(false)} />}
    <nav className={"sidebar" + (menuOpen ? " open" : "")} aria-label="Main">
      <div className="brand">
        {brand}
        <p>Species identification from environmental DNA</p>
      </div>
      <div className="nav" ref={navRef}>
        <div className="nav-indicator" aria-hidden="true"
          style={box ? { transform: `translateY(${box.y}px)`, height: box.h } : { opacity: 0 }} />
        {Object.entries(PAGES).map(([k, v]) => <React.Fragment key={k}>
          {v.section && <div className="nav-section">{v.section}</div>}
          <button className={"navlink" + (k === active ? " active" : "")} aria-current={k === active ? "page" : undefined} onClick={() => go(k)}>
            <NavIcon name={k} />{v.title}</button>
        </React.Fragment>)}
      </div>
      <div className="sidebar-foot">
        <button className="navlink" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
          <NavIcon name="theme" />{theme === "dark" ? "Light Mode" : "Dark Mode"}</button>
        <a className="navlink" href="/docs" target="_blank" rel="noreferrer"><NavIcon name="docs" />API Docs ↗</a>
      </div>
    </nav>
    <div className="main">
      <header className="topbar">
        <div key={route.view + (route.arg || "")} className="page-anim"><h2>{cur.title}</h2><div className="sub">{cur.sub}</div></div>
        <div className="topbar-actions">
          <button className="assist-btn" aria-expanded={chatOpen}
            onClick={() => chatOpen ? setChatOpen(false) : openChat(route.view === "run" ? route.arg : null)}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M5 18l-1 3 4-2h9a3 3 0 0 0 3-3V7a3 3 0 0 0-3-3H7a3 3 0 0 0-3 3v9" /></svg>
            Assistant</button>
        </div>
      </header>
      <main className="content">
        <div key={route.view + (route.arg || "")} className="page-anim">
          <Boundary viewKey={route.view + (route.arg || "")}>
            <View go={go} openChat={openChat} runId={route.arg} ids={route.arg} />
          </Boundary>
        </div>
      </main>
    </div>
    <AX.ChatDrawer open={chatOpen} onClose={() => setChatOpen(false)} runId={chatRun} setRunId={setChatRun} />
  </div>;
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
