/* Analysis pages of the main web app: upload a sample, follow its run, read
   its findings, browse/compare run history -- and the project assistant
   (chat drawer) that app.jsx shows on every page.

   Loaded as a second module of web/index.html, before app.jsx, and handed to
   it through window.EdnaAnalysis, so the existing dashboard code stays as it
   was. Everything renders inside an .ax wrapper, which scopes these styles
   away from the dashboard's own (index.html, "analysis pages" section).

   Charts follow the same data-viz mark specs as the dashboard. Confidence
   decisions are ORDERED, so they use one blue ramp (dark = confident) rather
   than categorical hues; both modes' ramps were validated with --ordinal. */

const { useState, useEffect, useMemo, useCallback, useRef } = React;
const { CountUp } = window.EdnaUI;                                 // web/ui.jsx

/* ================================================================ api === */

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch (_) {}
  if (!r.ok) {
    const e = new Error((body && body.detail) || `HTTP ${r.status}`);
    e.status = r.status;
    throw e;
  }
  return body;
}
const postJSON = (path, data) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

/* fetch() cannot report upload progress; XHR can. */
function uploadFile(file, marker, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const fd = new FormData();
    fd.append("file", file);
    fd.append("marker", marker);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch (_) {}
      if (xhr.status >= 200 && xhr.status < 300) resolve({ status: xhr.status, body });
      else reject(new Error((body && body.detail) || `HTTP ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error("network error -- is the API running?"));
    xhr.open("POST", "/uploads");
    xhr.send(fd);
  });
}

/* The pre-upload duplicate check is an optimisation, never a gate. It hashes
   the file in 8 MB slices with hash-wasm, which works on any origin and never
   holds the whole file in memory; WebCrypto is the fallback (it only exists on
   secure origins such as localhost, not on e.g. http://0.0.0.0). If neither
   is available the check is skipped -- the server's own duplicate check still
   applies after upload. */
async function sha256Hex(file) {
  try {
    if (window.hashwasm) {
      const h = await hashwasm.createSHA256();
      h.init();
      const step = 8 * 2 ** 20;
      for (let i = 0; i < file.size; i += step) h.update(new Uint8Array(await file.slice(i, i + step).arrayBuffer()));
      return h.digest("hex");
    }
    if (window.crypto && crypto.subtle) {
      const buf = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
      return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, "0")).join("");
    }
  } catch (_) { /* fall through: skip the pre-check */ }
  return null;
}

/* `path` null = idle. `tick` re-fetches. */
function useApi(path, tick = 0) {
  const [s, set] = useState({ data: null, loading: !!path, error: null });
  useEffect(() => {
    if (!path) { set({ data: null, loading: false, error: null }); return; }
    let alive = true;
    set((p) => ({ ...p, loading: true, error: null }));
    api(path).then((d) => alive && set({ data: d, loading: false, error: null }))
      .catch((e) => alive && set({ data: null, loading: false, error: e.message }));
    return () => { alive = false; };
  }, [path, tick]);
  return s;
}

/* ============================================================ helpers === */

const fmtInt = (n) => (n === null || n === undefined ? "—" : Number(n).toLocaleString());
const fmtPct = (n, d = 1) => (n === null || n === undefined ? "—" : `${Number(n).toFixed(d)}%`);
const fmtNum = (n, d = 3) => (n === null || n === undefined ? "—" : Number(n).toFixed(d));
const fmtBytes = (b) => b < 1024 ? `${b} B` : b < 2 ** 20 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 2 ** 20).toFixed(1)} MB`;
const fmtWhen = (iso) => { try { return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }); } catch (_) { return iso; } };
const fmtSecs = (s) => s === null || s === undefined ? "—" : s < 90 ? `${Math.round(s)} s` : `${(s / 60).toFixed(1)} min`;
/* Data labels ("fish_16s_mitofish") shown as Title Case words. */
const humanize = (s) => String(s).replace(/_/g, " ").replace(/(^|[\s(/·-])([a-z])/g, (_, p, c) => p + c.toUpperCase());
/* Older stored summaries spell thresholds in ASCII. */
const tidy = (s) => cap(String(s || "").replace(/>=/g, "≥").replace(/<=/g, "≤").replace(/ -- /g, " — "));
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

const DECISIONS = {
  assigned:                  { label: "Assigned",             color: "var(--d-assigned)" },
  best_guess:                { label: "Best guess",           color: "var(--d-best)" },
  best_guess_low_confidence: { label: "Low-confidence guess", color: "var(--d-low)" },
  unclassified:              { label: "Unclassified",         color: "var(--d-none)" },
  no_primer:                 { label: "No primer",            color: "var(--d-noprimer)" },
};
const decColor = (d) => (DECISIONS[d] || {}).color || "var(--axis)";
const decLabel = (d) => (DECISIONS[d] || {}).label || humanize(d);
const SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
const go = (path) => { location.hash = "#" + path; };

/* ============================================================== icons === */

const Icon = ({ d, size = 18, stroke = 1.7, children, ...rest }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={stroke}
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...rest}>{d ? <path d={d} /> : children}</svg>
);
const I = {
  upload: (p) => <Icon {...p}><path d="M12 16V4M7 9l5-5 5 5" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" /></Icon>,
  file: (p) => <Icon {...p}><path d="M14 3H7a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V7z" /><path d="M14 3v4h4" /></Icon>,
  chat: (p) => <Icon {...p}><path d="M5 18l-1 3 4-2h9a3 3 0 0 0 3-3V7a3 3 0 0 0-3-3H7a3 3 0 0 0-3 3v9" /></Icon>,
  send: (p) => <Icon {...p}><path d="M5 12h13M13 6l6 6-6 6" /></Icon>,
  close: (p) => <Icon {...p} d="M6 6l12 12M18 6L6 18" />,
  sun: (p) => <Icon {...p}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></Icon>,
  moon: (p) => <Icon {...p} d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />,
  download: (p) => <Icon {...p}><path d="M12 4v12M7 11l5 5 5-5" /><path d="M5 20h14" /></Icon>,
  trash: (p) => <Icon {...p}><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" /></Icon>,
  check: (p) => <Icon {...p} d="M5 12l5 5 9-10" />,
  alert: (p) => <Icon {...p}><path d="M12 3l9 16H3z" /><path d="M12 10v4M12 17h.01" /></Icon>,
  info: (p) => <Icon {...p}><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></Icon>,
  copy: (p) => <Icon {...p}><rect x="8" y="8" width="12" height="12" rx="2" /><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3" /></Icon>,
  plus: (p) => <Icon {...p} d="M12 5v14M5 12h14" />,
  arrow: (p) => <Icon {...p} d="M5 12h14M13 6l6 6-6 6" />,
  spark: (p) => <Icon {...p}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M6 18l2.5-2.5M15.5 8.5L18 6" /></Icon>,
};

const Helix = (props) => (
  <svg viewBox="0 0 420 300" fill="none" stroke="#fff" strokeWidth="2" {...props} aria-hidden="true">
    {Array.from({ length: 13 }).map((_, i) => {
      const x = 20 + i * 30, a = Math.sin(i * 0.62) * 90;
      return <g key={i}>
        <line x1={x} y1={150 - a} x2={x} y2={150 + a} strokeOpacity=".5" />
        <circle cx={x} cy={150 - a} r="6" fill="#fff" fillOpacity=".9" stroke="none" />
        <circle cx={x} cy={150 + a} r="6" fill="#fff" fillOpacity=".45" stroke="none" />
      </g>;
    })}
    <path d={"M20 150 " + Array.from({ length: 121 }).map((_, k) => { const x = 20 + k * 3; return `L${x} ${150 - Math.sin((x - 20) / 30 * 0.62) * 90}`; }).join(" ")} />
    <path d={"M20 150 " + Array.from({ length: 121 }).map((_, k) => { const x = 20 + k * 3; return `L${x} ${150 + Math.sin((x - 20) / 30 * 0.62) * 90}`; }).join(" ")} strokeOpacity=".6" />
  </svg>
);

const HelixMark = ({ className }) => {
  const p = window.EDNAFX && window.EDNAFX.helixPaths ? window.EDNAFX.helixPaths(120, 240, 6) : { a: "", b: "", rungs: "" };
  return <svg className={className} viewBox="0 0 120 240" fill="none" stroke="currentColor" strokeLinecap="round" aria-hidden="true">
    <path d={p.rungs} strokeWidth="2" /><path d={p.b} strokeWidth="3" opacity=".7" /><path d={p.a} strokeWidth="3.4" /></svg>;
};

/* ============================================================ tooltip === */

const useTooltip = window.EdnaUI.useTooltip;    // rendered at page level (web/ui.jsx)
const TipRows = ({ title, rows }) => <div>
  {title && <div style={{ fontWeight: 600, marginBottom: 4 }}>{title}</div>}
  {rows.map(([k, v]) => <div className="t-row" key={k}><span className="t-key">{k}</span><span className="t-val">{v}</span></div>)}
</div>;

const Legend = ({ items }) => items.length < 2 ? null : <div className="legend">
  {items.map((it) => <span key={it.label}><i style={{ background: it.color }} />{it.label}</span>)}</div>;

/* ============================================================= charts === */

/* Charts draw at the container's real pixel width, so 1 SVG unit = 1 px and
   text stays at its intended size in a wide card, a half-width card or a phone. */
function useWidth(fallback = 720) {
  const ref = useRef(null);
  const [w, setW] = useState(fallback);
  const [inView, setInView] = useState(!!window.EdnaUI.REDUCED);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.floor(e.contentRect.width))));
    ro.observe(ref.current);
    let io;
    if (!inView) {   // draw in once on entering the viewport, then stay still
      io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setInView(true); io.disconnect(); } }, { threshold: 0.25 });
      io.observe(ref.current);
    }
    return () => { ro.disconnect(); io && io.disconnect(); };
  }, []);
  return [ref, w, inView];
}
const chartClass = (inView) => "fx-chart" + (inView ? " in" : "");

/* A bar rounded only at its data end, square at the baseline. */
function vbarPath(x, y, w, h, r = 4) {
  r = Math.min(r, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}
function hbarPath(x, y, w, h, r = 4) {
  r = Math.min(r, h / 2, w);
  return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
}
const niceMax = (v) => { if (v <= 0) return 1; const p = 10 ** Math.floor(Math.log10(v)); const m = v / p; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p; };

/* Vertical bars. items: [{label, value, color, tip}] */
function VBars({ items, height = 230, yLabel, threshold, valueFmt = fmtInt }) {
  const { show, hide, node } = useTooltip();
  const [ref, W, inView] = useWidth();
  const max0 = niceMax(Math.max(1, ...items.map((i) => i.value)));
  // Left margin sized to the widest tick label, so large counts are never clipped.
  const H = height, padL = Math.max(40, valueFmt(max0).length * 7 + 16), padR = 10, padT = 14, padB = 34;
  const iw = W - padL - padR, ih = H - padT - padB;
  const max = niceMax(Math.max(1, ...items.map((i) => i.value)));
  const y = (v) => padT + ih - (v / max) * ih;
  const bw = iw / items.length;
  const ticks = 4;
  const every = Math.ceil(items.length / Math.max(4, Math.floor(iw / 34)));
  return <div ref={ref} className={chartClass(inView)}>
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }} role="img" aria-label={yLabel}>
      {Array.from({ length: ticks + 1 }).map((_, i) => {
        const v = (max * i) / ticks;
        return <g key={i}>
          <line x1={padL} x2={W - padR} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
          <text x={padL - 8} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--ink-muted)">{valueFmt(v)}</text>
        </g>;
      })}
      {items.map((it, i) => {
        const x = padL + i * bw + 1, w = Math.max(1, bw - 2), h = Math.max(it.value > 0 ? 2 : 0, y(0) - y(it.value));
        return <g key={i}>
          <rect x={padL + i * bw} y={padT} width={bw} height={ih} fill="transparent"
            onMouseMove={(e) => show(e, it.tip || <TipRows title={it.label} rows={[["reads", valueFmt(it.value)]]} />)} onMouseLeave={hide} />
          {h > 0 && <path className="fx-bar" style={{ "--i": i }} d={vbarPath(x, y(0) - h, w, h)} fill={it.color} pointerEvents="none" />}
          {i % every === 0 && <text x={padL + i * bw + bw / 2} y={H - padB + 16} textAnchor="middle" fontSize="11" fill="var(--ink-muted)">{it.label}</text>}
        </g>;
      })}
      {threshold && (() => {
        const tx = padL + threshold.index * bw;
        return <g pointerEvents="none">
          <line x1={tx} x2={tx} y1={padT - 4} y2={y(0)} stroke="var(--ink)" strokeDasharray="3 3" strokeOpacity=".55" />
          <text x={tx - 6} y={padT + 8} textAnchor="end" fontSize="11" fill="var(--ink-2)">{threshold.label}</text>
        </g>;
      })()}
      <line x1={padL} x2={W - padR} y1={y(0)} y2={y(0)} stroke="var(--axis)" />
    </svg>
    {node}
  </div>;
}

/* Horizontal bars with direct value labels. items: [{label, value, color, extra:[[k,v]]}] */
function HBars({ items, valueFmt = fmtInt, color = "var(--accent)", labelWidth = 210 }) {
  const { show, hide, node } = useTooltip();
  const [ref, W, inView] = useWidth();
  if (!items.length) return <p className="muted" style={{ margin: 0 }}>Nothing to show.</p>;
  const rowH = 28, padL = Math.min(labelWidth, Math.round(W * 0.42)), padR = 64;
  const H = items.length * rowH + 4, iw = W - padL - padR;
  const max = Math.max(...items.map((i) => i.value)) || 1;
  const maxc = Math.floor((padL - 14) / 7);
  return <div ref={ref} className={chartClass(inView)}>
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }} role="img">
      {items.map((it, i) => {
        const yy = i * rowH + 4, bh = 16, w = Math.max(2, (it.value / max) * iw);
        const lab = it.label.length > maxc ? it.label.slice(0, maxc - 1) + "…" : it.label;
        return <g key={it.label + i} onMouseMove={(e) => show(e, <TipRows title={it.label} rows={[["value", valueFmt(it.value)], ...(it.extra || [])]} />)} onMouseLeave={hide}>
          <rect x="0" y={yy - 4} width={W} height={rowH} fill="transparent" />
          <text x={padL - 10} y={yy + 12} textAnchor="end" fontSize="12.5" fill="var(--ink-2)">{lab}</text>
          <path className="fx-hbar" style={{ "--i": i }} d={hbarPath(padL, yy, w, bh)} fill={it.color || color} />
          <text x={padL + w + 8} y={yy + 12} fontSize="12" fill="var(--ink-2)" fontFamily="var(--mono)">{valueFmt(it.value)}</text>
        </g>;
      })}
    </svg>
    {node}
  </div>;
}

/* 100% stacked horizontal bars, one row per entity. rows: [{label, segments:[{key,label,value,color}]}] */
function Stacked({ rows, showRowLabels = true, height = 26 }) {
  const { show, hide, node } = useTooltip();
  const [ref, inView] = window.EdnaUI.useInView();
  return <div ref={ref} className={chartClass(inView)} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
    {rows.map((r) => {
      const total = r.segments.reduce((a, s) => a + s.value, 0) || 1;
      const visible = r.segments.filter((s) => s.value > 0);
      return <div key={r.label}>
        {showRowLabels && <div style={{ fontSize: 13, marginBottom: 5, color: "var(--ink-2)" }}>{r.label}</div>}
        <div style={{ display: "flex", gap: 2, height, overflow: "hidden" }}>
          {visible.map((s, i) => <div key={s.key} className="fx-seg" style={{ flex: `${s.value} 1 0`, background: s.color, minWidth: 3, "--i": i, borderRadius: 4 }}
            onMouseMove={(e) => show(e, <TipRows title={s.label} rows={[["reads", fmtInt(s.value)], ["share", fmtPct(100 * s.value / total)]]} />)}
            onMouseLeave={hide} />)}
        </div>
      </div>;
    })}
    {node}
  </div>;
}

/* Grouped vertical bars. labels: [..], series: [{name, color, values}] */
function GroupedBars({ labels, series, valueFmt = (v) => fmtNum(v, 3), height = 250, domainMax }) {
  const { show, hide, node } = useTooltip();
  const [ref, W, inView] = useWidth();
  const H = height, padL = 50, padR = 10, padT = 12, padB = 46;
  const iw = W - padL - padR, ih = H - padT - padB;
  const max = domainMax || niceMax(Math.max(1e-9, ...series.flatMap((s) => s.values.map((v) => v || 0))));
  const y = (v) => padT + ih - (v / max) * ih;
  const gw = iw / labels.length, n = series.length;
  const bw = Math.min(40, (gw - 18 - (n - 1) * 2) / n);
  const maxc = Math.max(6, Math.floor(gw / 6.6));
  return <div ref={ref} className={chartClass(inView)}>
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }} role="img">
      {Array.from({ length: 5 }).map((_, i) => {
        const v = (max * i) / 4;
        return <g key={i}><line x1={padL} x2={W - padR} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
          <text x={padL - 8} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--ink-muted)">{valueFmt(v)}</text></g>;
      })}
      {labels.map((lab, gi) => {
        const x0 = padL + gi * gw + (gw - (bw * n + (n - 1) * 2)) / 2;
        const short = String(lab).length > maxc ? String(lab).slice(0, maxc - 1) + "…" : String(lab);
        return <g key={lab + gi}>
          {series.map((s, si) => {
            const v = s.values[gi] || 0, h = Math.max(v > 0 ? 2 : 0, y(0) - y(v));
            return h > 0 && <path key={s.name} className="fx-bar" d={vbarPath(x0 + si * (bw + 2), y(0) - h, bw, h)} fill={s.color} style={{ cursor: "pointer", "--i": gi * n + si }}
              onMouseMove={(e) => show(e, <TipRows title={String(lab)} rows={series.map((ss) => [ss.name, valueFmt(ss.values[gi])])} />)} onMouseLeave={hide} />;
          })}
          <text x={padL + gi * gw + gw / 2} y={H - padB + 17} textAnchor="middle" fontSize="11.5" fill="var(--ink-2)">{humanize(short)}</text>
        </g>;
      })}
      <line x1={padL} x2={W - padR} y1={y(0)} y2={y(0)} stroke="var(--axis)" />
    </svg>
    {node}
  </div>;
}

/* ============================================================== bits === */

const Pill = ({ status }) => <span className={`pill ${status}`}><i />{status}</span>;
const Dec = ({ d }) => <span className="dec"><i style={{ background: decColor(d) }} />{decLabel(d)}</span>;
const Tile = ({ label, value, foot }) => <div className="tile"><div className="label">{label}</div><div className="value"><CountUp value={value} /></div>{foot && <div className="foot">{foot}</div>}</div>;
const Card = ({ title, hint, right, children, style }) => <section className="card" style={style}>
  {(title || right) && <div className="card-head"><div>{title && <h2>{title}</h2>}{hint && <p className="hint">{hint}</p>}</div>{right}</div>}
  {children}
</section>;
const Notice = ({ kind = "", icon, children }) => <div className={`notice ${kind}`} role={kind === "err" ? "alert" : undefined}>
  {icon || (kind === "err" || kind === "warn" ? <I.alert /> : kind === "ok" ? <I.check /> : <I.info />)}<div>{children}</div></div>;
const Skel = ({ h = 120 }) => <div className="skel" style={{ height: h }} />;

function Ring({ pct, size = 92, label }) {
  const r = (size - 10) / 2, c = 2 * Math.PI * r;
  return <svg className="ring" width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`${Math.round(pct)}% ${label || ""}`}>
    <circle cx={size / 2} cy={size / 2} r={r} stroke="var(--surface-3)" strokeWidth="8" fill="none" />
    <circle cx={size / 2} cy={size / 2} r={r} stroke="var(--accent)" strokeWidth="8" fill="none" strokeLinecap="round"
      strokeDasharray={c} strokeDashoffset={c * (1 - Math.min(100, pct) / 100)} transform={`rotate(-90 ${size / 2} ${size / 2})`} />
    <text x="50%" y="50%" textAnchor="middle" dominantBaseline="central" fontSize="17" fontWeight="600" fill="var(--ink)" fontFamily="var(--mono)">{Math.round(pct)}%</text>
  </svg>;
}

/* Poll a run until it settles. */
function useRunPoll(runId) {
  const [run, setRun] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    if (!runId) { setRun(null); return; }
    let alive = true, timer = null;
    const tick = async () => {
      try {
        const r = await api(`/runs/${runId}`);
        if (!alive) return;
        setRun(r); setError(null);
        if (r.status === "queued" || r.status === "running") timer = setTimeout(tick, 1500);
      } catch (e) { if (alive) setError(e.message); }
    };
    tick();
    return () => { alive = false; clearTimeout(timer); };
  }, [runId]);
  return { run, error };
}

function RunProgress({ run }) {
  const total = run.progress_total || 0, done = run.progress_done || 0;
  const started = run.started_at ? new Date(run.started_at) : null;
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t); }, []);
  const elapsed = started ? (now - started) / 1000 : 0;
  const pct = run.progress_pct || 0;
  const eta = pct > 3 && elapsed > 5 ? elapsed * (100 - pct) / pct : null;
  return <div className="progress-wrap">
    <Ring pct={pct} label="classified" />
    <div style={{ minWidth: 0 }}>
      <div style={{ fontWeight: 600 }}>{run.status === "queued" ? "Queued — waiting for the worker" : "Classifying reads"}</div>
      <div className="muted" style={{ fontSize: 13.5, marginTop: 2 }}>
        {run.status === "running" && total ? <>
          <span className="mono">{fmtInt(done)} / {fmtInt(total)}</span> distinct sequences
          {" · "}{fmtSecs(elapsed)} elapsed{eta !== null && <> · about {fmtSecs(eta)} left</>}
        </> : run.marker === "auto" ? "Detecting the gene region from your reads…" : "Preparing reads and loading the reference index…"}
      </div>
      <div className="muted" style={{ fontSize: 12.5, marginTop: 6 }}>Identical reads are classified once, so the count is of distinct sequences.</div>
    </div>
  </div>;
}

/* ========================================================= analyze === */

function AnalyzeView({ openChat }) {
  const markers = useApi("/markers");
  const [file, setFile] = useState(null);
  const [over, setOver] = useState(false);
  const [phase, setPhase] = useState({ step: "idle" });
  const inputRef = useRef(null);
  const { run } = useRunPoll(phase.runId);
  // The headline counts and recent-runs table re-load whenever the run on this
  // page changes state (queued -> running -> done/failed), so they never go stale.
  const runs = useApi("/runs?limit=200", `${phase.runId || ""}:${run ? run.status : ""}`);

  const pick = (f) => { if (f) { setFile(f); setPhase({ step: "idle" }); } };
  const busy = ["hashing", "checking", "uploading"].includes(phase.step) || (run && ["queued", "running"].includes(run.status));

  async function start() {
    if (!file) return;
    try {
      setPhase({ step: "hashing" });
      const sha = await sha256Hex(file);
      if (sha) {
        setPhase({ step: "checking" });
        const chk = await postJSON("/uploads/check", { sha256: sha });
        if (chk.already_analysed) { setPhase({ step: "duplicate", existing: chk.runs[0] }); return; }
      }
      setPhase({ step: "uploading", pct: 0 });
      // "auto": the server decides the reference set from the reads themselves.
      const { status, body } = await uploadFile(file, "auto", (p) => setPhase({ step: "uploading", pct: p * 100 }));
      if (status === 200 && body.duplicate) { setPhase({ step: "duplicate", existing: body.run }); return; }
      setPhase({ step: "running", runId: body.run.id, nReads: body.n_reads });
    } catch (e) {
      // A server started before automatic detection existed rejects "auto".
      const stale = /unknown marker 'auto'/.test(e.message);
      setPhase({ step: "error", message: stale
        ? "The server is running an older version that cannot detect the reference set automatically. Restart it (stop uvicorn with Ctrl+C and start it again), then try again."
        : e.message });
    }
  }

  const done = (runs.data && runs.data.runs.filter((r) => r.status === "done")) || [];
  const totalReads = done.reduce((a, r) => a + (r.n_reads || 0), 0);
  const sets = markers.data ? markers.data.markers.filter((m) => m.available) : [];
  const detection = markers.data && markers.data.detection;

  return <div className="stack" style={{ gap: 22 }}>
    <section className="hero">
      <Helix className="hero-helix" />
      <div className="eyebrow">New analysis</div>
      <h1>Identify Species From A Water Sample's DNA</h1>
      <p>Upload sequencing reads and the production pipeline matches each one against a reference database — BM25 retrieval, MMR diversification, Smith-Waterman verification — and tells you how confident each call is.</p>
      <div className="hero-stats">
        <div className="hero-stat"><b>{runs.data ? <CountUp value={fmtInt(done.length)} /> : "—"}</b><span>Runs Analysed</span></div>
        <div className="hero-stat"><b>{runs.data ? <CountUp value={fmtInt(totalReads)} /> : "—"}</b><span>Reads Classified</span></div>
        <div className="hero-stat"><b>{markers.data ? <CountUp value={String(markers.data.markers.filter((m) => m.available).length)} /> : "—"}</b><span>Reference Sets Ready</span></div>
      </div>
    </section>

    <div className="grid cols-2" style={{ alignItems: "start" }}>
      <Card title="New analysis" hint="FASTQ or FASTA, plain or gzipped.">
        <div className="stack">
          {!file ? (
            <label className={`drop ${over ? "over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
              onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files[0]); }}>
              <input ref={inputRef} type="file" className="sr-only" accept=".fastq,.fq,.fasta,.fa,.fna,.gz,.txt"
                onChange={(e) => pick(e.target.files[0])} />
              <div className="drop-icon"><I.upload size={22} /></div>
              <strong>Drop a read file here</strong>
              <span className="muted" style={{ fontSize: 13.5 }}>Or Click To Browse · <span className="keep-case">.fastq · .fasta · .gz</span></span>
            </label>
          ) : (
            <div className="file-chip">
              <I.file size={22} style={{ color: "var(--accent)", flex: "0 0 auto" }} />
              <div style={{ minWidth: 0, flex: 1 }}>
                <div className="name">{file.name}</div>
                <div className="muted mono" style={{ fontSize: 12 }}>{fmtBytes(file.size)}</div>
              </div>
              <button className="btn sm" onClick={() => { setFile(null); setPhase({ step: "idle" }); }} disabled={busy}>Change</button>
            </div>
          )}

          <div>
            <div className="field" style={{ marginBottom: 10 }}>Reference set — detected from your reads</div>
            <div className="markers">
              {markers.loading && <Skel h={150} />}
              {markers.error && <Notice kind="err">{markers.error}</Notice>}
              {markers.data && markers.data.markers.map((m) => (
                <div key={m.key} className={`marker ${m.available ? "" : "disabled"}`} style={{ cursor: "default" }}>
                  <span aria-hidden="true" style={{ width: 8, height: 8, marginTop: 7, flex: "0 0 8px",
                    background: m.available ? "var(--ink-muted)" : "transparent", border: "1px solid var(--ink-muted)" }} />
                  <div>
                    <div className="t">{m.label}</div>
                    <div className="d">{m.primers ? `Primers ${m.primers}` : "No primer trimming"}
                      {m.n_references !== null && ` · ${fmtInt(m.n_references)} references`}
                      {m.validation && ` · ${m.validated_part}: ${fmtInt(m.validation.n_assigned)} of ${fmtInt(m.validation.n_reads)} reads assigned`}
                      {!m.available && " · reference files missing"}</div>
                  </div>
                </div>
              ))}
            </div>
            {detection && <p className="muted" style={{ fontSize: 12.5, margin: "10px 0 0" }}>
              Nothing to choose: the server compares your reads with each set (share of shared {detection.method.match(/\d+-mers/) ? detection.method.match(/\d+-mers/)[0] : "k-mers"})
              and uses the one they match, or flags the run if none fits.</p>}
          </div>

          <button className="btn primary block" onClick={start} disabled={!file || busy || !sets.length}>
            {busy ? "Working…" : <>Analyse sample <I.arrow size={16} /></>}
          </button>
        </div>
      </Card>

      <Card title="Status">
        <AnalyzeStatus phase={phase} run={run} file={file} marker={sets[0]} detection={detection} onReset={() => setPhase({ step: "idle" })} openChat={openChat} />
      </Card>
    </div>

    <Card title="Recent runs" right={<a className="btn sm" href="#runs">All history <I.arrow size={14} /></a>}>
      {runs.loading ? <Skel h={120} /> : <RunsTable runs={(runs.data ? runs.data.runs : []).slice(0, 5)} compact />}
    </Card>
  </div>;
}

function AnalyzeStatus({ phase, run, file, marker, detection, onReset, openChat }) {
  if (phase.step === "idle") return <div className="stack" style={{ gap: 12 }}>
    {[["1", "Duplicate check", "The file's SHA-256 is checked against your history first — a file already analysed is never re-run."],
      ["2", "Gene-region detection", detection
        ? `A sample of your reads is compared with every reference set (${detection.method}); the best match must share at least ${(detection.min_score * 100).toFixed(0)}% and ${detection.min_ratio_to_runner_up}× the runner-up, otherwise the run is flagged.`
        : "A sample of your reads is compared with every reference set to find the one they come from."],
      ["3", "Read preparation", "Reads are primer-trimmed and oriented as in the validated run for that set — or, if the primers are already removed, used as uploaded."],
      ["4", "Classification", `Each distinct sequence goes through BM25 → MMR → Smith-Waterman, then the ${marker ? marker.identity_threshold : "…"}% identity decision.`]].map(([n, t, d]) =>
      <div key={n} className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
        <span className="mono" style={{ width: 26, height: 26, display: "grid", placeItems: "center", border: "1px solid var(--border-strong)", color: "var(--ink)", fontSize: 13, flex: "0 0 26px" }}>{n}</span>
        <div><div style={{ fontWeight: 600, fontSize: 14 }}>{t}</div><div className="muted" style={{ fontSize: 13.5 }}>{d}</div></div>
      </div>)}
    {!file && <p className="muted" style={{ margin: "4px 0 0", fontSize: 13.5 }}>Choose a file to begin.</p>}
  </div>;

  if (phase.step === "hashing" || phase.step === "checking") return <div className="row"><span className="typing"><i /><i /><i /></span>
    <span>{phase.step === "hashing" ? "Fingerprinting the file…" : "Checking your history for this file…"}</span></div>;

  if (phase.step === "uploading") {
    const sent = phase.pct >= 99.9;
    return <div className="stack" style={{ gap: 10 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span>{sent ? "Checking" : "Uploading"} <b>{file && file.name}</b></span>
        {sent ? <span className="typing"><i /><i /><i /></span> : <span className="mono">{Math.round(phase.pct)}%</span>}</div>
      <div className="bar-track"><div className="bar-fill" style={{ width: `${phase.pct}%` }} /></div>
      <p className="muted" style={{ margin: 0, fontSize: 13 }}>{sent
        ? "Uploaded. The server is reading every record to validate the file and saving a copy — large files take a little while."
        : "The file is validated on the server before the analysis is queued."}</p>
    </div>;
  }

  if (phase.step === "duplicate") return <div className="stack" style={{ gap: 12 }}>
    <Notice kind="ok"><b>Already analysed — no upload needed.</b><br />This exact file was analysed
      as <b>{phase.existing.filename}</b> on {fmtWhen(phase.existing.created_at)}
      {phase.existing.marker && phase.existing.marker !== "auto" && <> against {humanize(phase.existing.marker)}</>}.</Notice>
    <div className="row">
      <a className="btn primary" href={`#run/${phase.existing.id}`}>Open those results <I.arrow size={16} /></a>
      <button className="btn" onClick={onReset}>Back</button>
    </div>
  </div>;

  if (phase.step === "error") return <div className="stack" style={{ gap: 12 }}>
    <Notice kind="err"><b>That didn't work.</b><br />{phase.message}</Notice>
    <div><button className="btn" onClick={onReset}>Try again</button></div>
  </div>;

  if (phase.step === "running") {
    if (!run) return <Skel h={92} />;
    if (run.status === "failed") return <Notice kind="err"><b>The run failed.</b><br />{run.error}</Notice>;
    if (run.status === "done") {
      const s = run.summary, det = s.marker_detection;
      return <div className="stack" style={{ gap: 14 }}>
        {det && !det.confident
          ? <Notice kind="warn"><b>Done, but no reference set fits these reads.</b> {cap(det.reason)}. The results are against the closest set and are unlikely to be meaningful.</Notice>
          : <Notice kind="ok"><b>Done in {fmtSecs(s.seconds)}.</b> {fmtInt(s.input.n_reads)} reads, {fmtPct(s.assignment_rate_pct)} assigned with confidence.</Notice>}
        <div className="muted" style={{ fontSize: 13.5 }}>Reference set: <b style={{ color: "var(--ink)" }}>{s.marker_label}</b>{det ? (det.confident ? " — detected automatically" : " — closest match") : ""}</div>
        <Stacked showRowLabels={false} rows={[{ label: "decisions", segments: s.decisions.map((d) => ({ key: d.decision, label: decLabel(d.decision), value: d.reads, color: decColor(d.decision) })) }]} />
        <Legend items={s.decisions.filter((d) => d.reads).map((d) => ({ label: `${decLabel(d.decision)} ${fmtInt(d.reads)}`, color: decColor(d.decision) }))} />
        <div className="row">
          <a className="btn primary" href={`#run/${run.id}`}>View full results <I.arrow size={16} /></a>
          <button className="btn" onClick={() => openChat(run.id)}><I.chat size={16} /> Ask about it</button>
        </div>
      </div>;
    }
    return <RunProgress run={run} />;
  }
  return null;
}

/* ========================================================== runs === */

const settled = (r) => !["queued", "running"].includes(r.status);

function RunsTable({ runs, compact, selected, onToggle, onDelete }) {
  if (!runs.length) return <div className="empty"><I.file size={34} /><div>No runs yet. Upload a file on the Analyse tab.</div></div>;
  return <div className="table-wrap" style={compact ? { maxHeight: "none" } : null}>
    <table className="t">
      <thead><tr>
        {onToggle && <th style={{ width: 36 }}><span className="sr-only">Select</span></th>}
        <th>File</th><th>Marker</th><th>Status</th><th className="num">Reads</th><th className="num">Assigned</th><th>Uploaded</th>
        {onDelete && <th style={{ width: 52 }}><span className="sr-only">Delete</span></th>}
      </tr></thead>
      <tbody>{runs.map((r) => (
        <tr key={r.id} className="click" onClick={() => go(`run/${r.id}`)}>
          {onToggle && <td onClick={(e) => e.stopPropagation()}>
            <input type="checkbox" aria-label={`Select ${r.filename}`} checked={selected.includes(r.id)} disabled={!settled(r)}
              onChange={() => onToggle(r.id)} style={{ width: 16, height: 16, accentColor: "var(--accent)" }} /></td>}
          <td><div className="keep-case" style={{ fontWeight: 500 }}>{r.filename}</div><div className="muted mono" style={{ fontSize: 11.5 }}>{r.id}</div></td>
          <td style={{ fontSize: 13 }}>{r.marker === "auto" ? <span className="muted">Detecting…</span> : humanize(r.marker)}
            {r.marker_fits === false && <div style={{ color: "var(--critical-ink)", fontSize: 12 }}>no matching set</div>}</td>
          <td><Pill status={r.status} /></td>
          <td className="num">{fmtInt(r.n_reads)}</td>
          <td className="num">{r.assignment_rate_pct === null || r.assignment_rate_pct === undefined ? "—" : fmtPct(r.assignment_rate_pct)}</td>
          <td style={{ fontSize: 13, whiteSpace: "nowrap" }}>{fmtWhen(r.created_at)}</td>
          {onDelete && <td onClick={(e) => e.stopPropagation()} style={{ textAlign: "right" }}>
            {settled(r) && <button className="icon-btn" aria-label={`Delete ${r.filename}`} title="Delete this run"
              onClick={() => onDelete([r])}><I.trash size={15} /></button>}</td>}
        </tr>))}
      </tbody>
    </table>
  </div>;
}

/* Confirms and performs deletion of one or more runs: here, and from the cloud
   backup. Deletes one run at a time so progress is visible, and reports what
   happened in the cloud, including deletions queued for the next backup. */
function DeleteDialog({ runs, onClose }) {
  const backup = useApi("/backup/status");
  const [state, setState] = useState({ phase: "confirm" });
  const cancelRef = useRef(null);
  useEffect(() => { cancelRef.current && cancelRef.current.focus(); }, []);
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape" && state.phase !== "working") onClose(state.phase === "done"); };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [state.phase]);
  const b = backup.data;
  const cloudLine = !b ? "Checking the cloud backup…"
    : !b.enabled ? "Cloud backup is switched off on this server, so only the copy here is deleted."
    : !b.configured ? `The cloud backup is not connected (${b.reason}). The run is deleted here, and from the cloud once it is connected.`
    : `It is also deleted from the cloud backup (${b.remote}) — its results, its uploaded file, and its entry in the backed-up history.`;

  async function go_() {
    const results = [];
    for (let i = 0; i < runs.length; i++) {
      setState({ phase: "working", i });
      try { results.push({ run: runs[i], res: await api(`/runs/${runs[i].id}`, { method: "DELETE" }) }); }
      catch (e) { results.push({ run: runs[i], error: e.message }); }
    }
    setState({ phase: "done", results });
  }

  const n = runs.length, many = n > 1;
  return <div className="modal-backdrop" role="presentation" onClick={() => state.phase !== "working" && onClose(state.phase === "done")}>
    <div className="modal" role="dialog" aria-modal="true" aria-labelledby="del-title" onClick={(e) => e.stopPropagation()}>
      {state.phase === "confirm" && <>
        <h2 id="del-title">delete {many ? `${n} runs` : "this run"}?</h2>
        <ul className="del-list">{runs.map((r) => <li key={r.id}><b>{r.filename}</b> <span className="muted mono">{r.id}</span></li>)}</ul>
        <p>Its per-read results and history entry are removed, and the uploaded file too unless another run still uses it.</p>
        <p className="muted">{cloudLine}</p>
        <p className="muted">This cannot be undone.</p>
        <div className="row" style={{ justifyContent: "flex-end", marginTop: 8 }}>
          <button ref={cancelRef} className="btn" onClick={() => onClose(false)}>Cancel</button>
          <button className="btn danger" onClick={go_} disabled={!b}><I.trash size={15} /> Delete {many ? `${n} runs` : "run"}</button>
        </div>
      </>}
      {state.phase === "working" && <div className="stack" style={{ gap: 14 }}>
        <h2 id="del-title">deleting…</h2>
        <div className="row"><span className="typing"><i /><i /><i /></span>
          <span>Run {state.i + 1} of {n}: <b>{runs[state.i].filename}</b> — removing it here and from the cloud</span></div>
        <div className="bar-track"><div className="bar-fill" style={{ width: `${(100 * state.i) / n}%` }} /></div>
      </div>}
      {state.phase === "done" && (() => {
        const ok = state.results.filter((x) => x.res), failed = state.results.filter((x) => x.error);
        const queued = ok.filter((x) => x.res.cloud && x.res.cloud.queued && x.res.cloud.queued.length);
        const skipped = ok.some((x) => x.res.cloud && x.res.cloud.skipped);
        return <div className="stack" style={{ gap: 14 }}>
          <h2 id="del-title">{failed.length ? "some runs were not deleted" : `${ok.length === 1 ? "run" : `${ok.length} runs`} deleted`}</h2>
          {ok.length > 0 && <Notice kind={queued.length ? "warn" : "ok"}>
            Deleted here: {ok.map((x) => x.run.filename).join(", ")}.{" "}
            {skipped ? "Cloud backup is off, so nothing was removed from the cloud."
              : queued.length ? <>Removing {queued.length === 1 ? "it" : "some"} from the cloud failed for now ({queued[0].res.cloud.error}); the next backup will finish the job.</>
              : "Also removed from the cloud backup."}</Notice>}
          {failed.map((x) => <Notice key={x.run.id} kind="err"><b>{x.run.filename}</b>: {x.error}</Notice>)}
          <div className="row" style={{ justifyContent: "flex-end" }}><button className="btn primary" onClick={() => onClose(true)}>Done</button></div>
        </div>;
      })()}
    </div>
  </div>;
}

/* Cloud backup of the run history (src/cloud_backup.py): where it goes, when
   it last succeeded, and a manual trigger. Polls while a backup is running. */
function BackupStatus() {
  const [tick, setTick] = useState(0);
  const st = useApi("/backup/status", tick);
  const [msg, setMsg] = useState(null);
  const d = st.data;
  useEffect(() => {
    if (!d || !d.running) return;
    const t = setTimeout(() => setTick((x) => x + 1), 3000);
    return () => clearTimeout(t);
  }, [d]);
  if (!d) return <span />;
  const last = d.last;
  const line = !d.configured ? `Cloud backup not connected — ${d.reason}`
    : d.running ? "Backing up to the cloud…"
    : last && last.ok ? `Backed up to ${d.remote} · ${fmtWhen(last.finished_at)}`
    : last && !last.ok ? `Last backup failed: ${last.error}${last.last_success_at ? ` (last good: ${fmtWhen(last.last_success_at)})` : ""}`
    : `Connected to ${d.remote} · not backed up yet`;
  const bad = !d.configured || (last && !last.ok && !d.running);
  async function now() {
    setMsg(null);
    try { await postJSON("/backup", {}); setTick((x) => x + 1); }
    catch (e) { setMsg(e.message); }
  }
  return <div className="row" style={{ gap: 14, maxWidth: 760 }}>
    <span className="dec" style={{ whiteSpace: "normal", color: bad ? "var(--critical-ink)" : "var(--ink-2)" }}>
      <i style={{ background: bad ? "var(--critical)" : d.running ? "var(--ink)" : "var(--good)" }} />{line}</span>
    {d.configured && d.enabled && <button className="btn sm" onClick={now} disabled={d.running}>Back up now</button>}
    {msg && <span className="muted" style={{ fontSize: 13 }}>{msg}</span>}
  </div>;
}

function RunsView() {
  const [marker, setMarker] = useState("");
  const [status, setStatus] = useState("");
  const [sel, setSel] = useState([]);
  const [tick, setTick] = useState(0);
  const [deleting, setDeleting] = useState(null);       // runs awaiting confirmation
  const q = new URLSearchParams({ limit: 500, ...(marker && { marker }), ...(status && { status }) });
  const runs = useApi(`/runs?${q}`, tick);
  const markers = useApi("/markers");
  const toggle = (id) => setSel((s) => s.includes(id) ? s.filter((x) => x !== id) : [...s, id]);
  const all = runs.data ? runs.data.runs : [];
  const chosen = all.filter((r) => sel.includes(r.id));
  const canCompare = chosen.length >= 2 && chosen.length <= 10 && chosen.every((r) => r.status === "done");
  return <div className="stack">
    <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
      <BackupStatus />
      <div className="row">
        <button className="btn" disabled={!chosen.length} onClick={() => setDeleting(chosen)}>
          <I.trash size={15} /> Delete {chosen.length ? `${chosen.length} selected` : "selected"}</button>
        <button className="btn primary" disabled={!canCompare} onClick={() => go(`compare/${sel.join(",")}`)}
          title={canCompare ? "" : "Select 2–10 finished runs to compare"}>
          Compare {canCompare ? chosen.length : ""} runs <I.arrow size={16} /></button>
      </div>
    </div>
    {deleting && <DeleteDialog runs={deleting} onClose={(changed) => {
      setDeleting(null);
      if (changed) { setSel((s) => s.filter((id) => !deleting.some((r) => r.id === id))); setTick((t) => t + 1); }
    }} />}
    <Card title={runs.data ? `${fmtInt(runs.data.total)} run${runs.data.total === 1 ? "" : "s"}` : "Runs"}
      hint="Select runs to compare (2–10 finished runs) or delete them — here and from the cloud backup. A file already analysed is never re-run."
      right={<div className="row">
        <select className="input" aria-label="Filter by marker" value={marker} onChange={(e) => setMarker(e.target.value)}>
          <option value="">All markers</option>
          {markers.data && markers.data.markers.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
        </select>
        <select className="input" aria-label="Filter by status" value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="">Any status</option>{["done", "running", "queued", "failed"].map((s) => <option key={s} value={s}>{cap(s)}</option>)}
        </select>
      </div>}>
      {runs.loading ? <Skel h={220} /> : runs.error ? <Notice kind="err">{runs.error}</Notice> :
        <RunsTable runs={runs.data.runs} selected={sel} onToggle={toggle} onDelete={(rs) => setDeleting(rs)} />}
    </Card>
  </div>;
}

/* ======================================================= run detail === */

/* Where the run actually is: queued -> detecting the gene region (auto
   marker, nothing classified yet) -> classifying -> complete / failed.
   Every step shown is a state the server has reported; nothing is simulated. */
function runStage(run) {
  if (run.status === "queued") return 0;
  if (run.status === "running") return run.marker === "auto" || !run.progress_total ? 1 : 2;
  return 3;
}
function RunSteps({ run }) {
  const stage = runStage(run), failed = run.status === "failed";
  const steps = ["Queued", "Detecting Region", "Classifying", failed ? "Failed" : "Complete"];
  return <div className="run-steps" role="list" aria-label="Run progress">
    {steps.map((label, i) => {
      const done = i < stage || (i === 3 && run.status === "done");
      const cls = failed && i === 3 ? "failed" : done ? "done" : i === stage ? "current" : "";
      return <React.Fragment key={label}>
        {i > 0 && <span className={"run-link" + (i <= stage ? " done" : "")} style={{ "--d": `${i * 140}ms` }} />}
        <span className={"run-step " + cls} role="listitem" aria-current={i === stage && !done ? "step" : undefined}>
          <span className="node">{done && <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path className="check-draw" d="M5 12l5 5 9-10" /></svg>}
            {failed && i === 3 && <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18" /></svg>}</span>
          {label}</span>
      </React.Fragment>;
    })}
  </div>;
}

/* Download with immediate confirmation that something happened. */
function DownloadButton({ href, children }) {
  const [done, setDone] = useState(false);
  useEffect(() => { if (!done) return; const t = setTimeout(() => setDone(false), 2200); return () => clearTimeout(t); }, [done]);
  return <a className={"btn" + (done ? " ok" : "")} href={href} onClick={() => setDone(true)}>
    {done ? <><I.check size={16} /> Download Started</> : children}</a>;
}

function RunView({ runId, openChat }) {
  const { run, error } = useRunPoll(runId);
  const [deleting, setDeleting] = useState(false);
  if (error) return <Notice kind="err">{error}</Notice>;
  if (!run) return <div className="stack"><Skel h={70} /><Skel h={110} /><Skel h={300} /></div>;

  const s = run.summary;
  return <div className="stack">
    <div className="page-head">
      <HelixMark className="run-watermark" />
      <div style={{ minWidth: 0 }}>
        <div className="eyebrow"><a href="#runs" style={{ color: "inherit" }}>Runs</a> / <span className="mono">{run.id}</span></div>
        <h1 className="keep-case" style={{ overflowWrap: "anywhere" }}>{run.filename}</h1>
        <div className="row" style={{ marginTop: 8 }}>
          <Pill status={run.status} />
          <span className="muted" style={{ fontSize: 13.5 }}>{s ? s.marker_label : run.marker === "auto" ? "Reference set being detected" : cap(humanize(run.marker))} · Uploaded {fmtWhen(run.created_at)} · <span className="mono">{run.file_format}</span> · {fmtBytes(run.size_bytes)}</span>
        </div>
        <div style={{ marginTop: 16 }}><RunSteps run={run} /></div>
      </div>
      <div className="row">
        {run.status === "done" && <DownloadButton href={`/runs/${run.id}/download`}><I.download size={16} /> CSV</DownloadButton>}
        <button className="btn" onClick={() => openChat(run.id)}><I.chat size={16} /> Ask About This Run</button>
        {!["queued", "running"].includes(run.status) &&
          <button className="btn" onClick={() => setDeleting(true)} aria-label="Delete run"><I.trash size={16} /></button>}
      </div>
    </div>

    {deleting && <DeleteDialog runs={[run]} onClose={(changed) => { setDeleting(false); if (changed) go("runs"); }} />}
    {["queued", "running"].includes(run.status) && <Card><RunProgress run={run} /></Card>}
    {run.status === "failed" && <Notice kind="err"><b>This run failed.</b><br />{run.error}</Notice>}
    {run.status === "done" && s && <RunFindings run={run} s={s} />}
  </div>;
}

function RunFindings({ run, s }) {
  const assigned = s.decisions.find((d) => d.decision === "assigned") || { reads: 0 };
  const analysed = s.input.n_reads - ((s.decisions.find((d) => d.decision === "no_primer") || {}).reads || 0);
  const ofAnalysed = s.assignment_rate_of_analysed_pct ?? (analysed ? 100 * assigned.reads / analysed : null);
  const hist = s.identity.histogram;
  const firstNonEmpty = Math.max(0, hist.findIndex((b) => b.reads > 0) - 1);
  const shown = hist.slice(Math.min(firstNonEmpty, 10));        // skip the empty low bins, but keep the 50-100 span
  const thresholdIdx = shown.findIndex((b) => b.bin_start >= s.identity_threshold);
  const binColor = (b) => b.bin_start >= s.identity_threshold ? decColor("assigned")
    : b.bin_start >= s.best_guess_min_identity ? decColor("best_guess") : decColor("best_guess_low_confidence");
  const hub = s.references.top10_share_of_assigned_pct;
  const lineage = s.taxa.lineage_level_1_assigned;

  return <>
    <div className="grid cols-tiles">
      <Tile label="Reads" value={fmtInt(s.input.n_reads)} foot={`${fmtInt(s.input.n_unique_sequences_classified)} distinct sequences`} />
      <Tile label={`Assigned (≥ ${s.identity_threshold}%)`} value={fmtPct(s.assignment_rate_pct)}
        foot={s.input.primer_detection ? `${fmtPct(ofAnalysed)} of primer-matched reads` : `${fmtInt(assigned.reads)} reads`} />
      <Tile label="Median identity" value={fmtPct(s.identity.median_assigned)} foot={`Assigned Reads · All: ${fmtPct(s.identity.median_all_classified)}`} />
      <Tile label="Taxa detected" value={fmtInt(s.taxa.n_distinct_assigned)} foot="Distinct, Among Assigned Reads" />
      <Tile label="Runtime" value={fmtSecs(s.seconds)} foot={`${fmtInt(s.reference.n_references)} references searched`} />
    </div>

    {s.marker_detection && <DetectionCard det={s.marker_detection} chosen={s.marker} />}

    <Card title="Confidence decisions" hint="Share of all reads in each tier. Darker means a more confident call.">
      <div className="stack" style={{ gap: 14 }}>
        <Stacked showRowLabels={false} height={34} rows={[{ label: "decisions", segments: s.decisions.map((d) => ({ key: d.decision, label: decLabel(d.decision), value: d.reads, color: decColor(d.decision) })) }]} />
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(200px, 100%), 1fr))", gap: 10 }}>
          {s.decisions.map((d) => <div key={d.decision} style={{ padding: "10px 12px", border: "1px solid var(--border)" }}>
            <Dec d={d.decision} />
            <div className="mono" style={{ fontSize: 20, marginTop: 4 }}>{fmtInt(d.reads)} <span className="muted" style={{ fontSize: 13 }}>{fmtPct(d.pct_of_reads)}</span></div>
            <div className="muted" style={{ fontSize: 12 }}>{tidy(d.meaning)}</div>
          </div>)}
        </div>
      </div>
    </Card>

    <div className="grid cols-2">
      <Card title="Identity to nearest reference" hint={`Reads per ${s.identity.histogram_bin_width}-point bin. Reads at ≥ ${s.identity_threshold}% are assigned.`}>
        <VBars yLabel="reads" items={shown.map((b) => ({ label: `${b.bin_start}`, value: b.reads, color: binColor(b),
          tip: <TipRows title={`${b.bin_start}–${b.bin_end}% identity`} rows={[["reads", fmtInt(b.reads)]]} /> }))}
          threshold={thresholdIdx >= 0 ? { index: thresholdIdx, label: `${s.identity_threshold}% bar` } : null} />
      </Card>
      <Card title="Most-detected taxa" hint="Assigned reads only, by nearest reference organism.">
        <HBars items={s.taxa.top_assigned.slice(0, 10).map((t) => ({ label: t.taxon, value: t.reads,
          extra: [["share of assigned", fmtPct(t.pct_of_subset)], ["mean identity", fmtPct(t.mean_identity, 2)]] }))} />
      </Card>
    </div>

    <div className="grid cols-2">
      {lineage.length > 0 && <Card title="Lineage composition" hint="Assigned reads by taxonomy level 1 (phylum for bacteria; a supergroup for eukaryotes).">
        <HBars items={lineage.slice(0, 10).map((t) => ({ label: t.taxon, value: t.reads, extra: [["share", fmtPct(t.pct_of_assigned)]] }))} color="var(--s1)" />
      </Card>}
      <Card title="Input & references">
        <div className="stack" style={{ gap: 12 }}>
          <table className="t"><tbody>
            <tr><td>Read length (min / median / max)</td><td className="num">{s.input.read_length ? `${s.input.read_length.min} / ${s.input.read_length.median} / ${s.input.read_length.max} bp` : "—"}</td></tr>
            <tr><td>Read preparation</td><td className="num" style={{ whiteSpace: "normal", maxWidth: 360 }}>{s.input.preparation ? cap(s.input.preparation) : (s.input.primers ? `Primer Trim (${s.input.primers})` : "Reads Used As Uploaded")}</td></tr>
            {s.input.primer_detection && <tr><td>Primer found / not found</td><td className="num">{fmtInt(s.input.primer_detection.primer_found)} / {fmtInt(s.input.primer_detection.no_primer)}</td></tr>}
            <tr><td>Reads reaching the classifier</td><td className="num">{fmtInt(analysed)}</td></tr>
            <tr><td>Distinct references hit (assigned)</td><td className="num">{fmtInt(s.references.distinct_hit_by_assigned)}</td></tr>
            <tr><td>Top-10 references' share of assigned</td><td className="num">{fmtPct(hub)}</td></tr>
          </tbody></table>
          {s.references.concentrated && <Notice kind="warn">
            <b>Concentrated hits.</b> {fmtPct(hub)} of assigned reads land on the top 10 references (flagged at ≥ {fmtPct(s.references.hubness_flag_pct, 0)}).
            That can be a genuinely dominant species, or <i>hubness</i> — a few references absorbing unrelated reads (Explanation.md, Parts 2–3).
            {" "}Median identity of the assigned reads is {fmtPct(s.identity.median_assigned)}
            {s.identity.median_assigned >= s.identity_threshold + (100 - s.identity_threshold) / 2
              ? ", well above the bar, which points to real matches; check the read table to be sure."
              : ", close to the bar, so treat these calls with care and check the read table."}</Notice>}
        </div>
      </Card>
    </div>

    <ReadsExplorer run={run} />
  </>;
}

/* How the reference set was chosen: every set's share of the reads' k-mers. */
function DetectionCard({ det, chosen }) {
  return <Card title="Reference set detection"
    hint={`${det.sample_size.toLocaleString()} distinct reads compared with every reference set — share of their ${det.k}-mers found in each.`}>
    <div className="stack" style={{ gap: 14 }}>
      {!det.confident && <Notice kind="warn"><b>No reference set fits these reads.</b> {cap(det.reason)}. The run used the
        closest set, so its matches are unlikely to be real — the reads may come from a gene region none of these sets covers.</Notice>}
      <HBars valueFmt={(v) => `${v.toFixed(1)}%`} labelWidth={300}
        items={det.evidence.map((e) => ({ label: e.label, value: e.score * 100,
          color: e.marker === chosen && det.confident ? "var(--ink)" : "var(--axis)",
          extra: [["primers found", e.primer_fraction === null ? "n/a (no primers)" : fmtPct(e.primer_fraction * 100)],
                  ["reads prepared", e.mode === "primer_trim" ? "primer-trimmed" : "as uploaded"]] }))} />
      {det.confident && <p className="muted" style={{ margin: 0, fontSize: 13.5 }}>Chosen automatically — {det.reason}.</p>}
    </div>
  </Card>;
}

function ReadsExplorer({ run }) {
  const [decision, setDecision] = useState("");
  const [minId, setMinId] = useState("");
  const [taxon, setTaxon] = useState("");
  const [debTaxon, setDebTaxon] = useState("");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState(null);
  const per = 50;
  useEffect(() => { const t = setTimeout(() => setDebTaxon(taxon), 300); return () => clearTimeout(t); }, [taxon]);
  useEffect(() => setPage(0), [decision, minId, debTaxon]);
  const q = new URLSearchParams({ limit: per, offset: page * per, ...(decision && { decision }), ...(minId !== "" && { min_identity: minId }), ...(debTaxon && { taxon: debTaxon }) });
  const res = useApi(`/runs/${run.id}/reads?${q}`);
  const present = run.summary.decisions.filter((d) => d.reads > 0).map((d) => d.decision);
  const total = res.data ? res.data.total_matching : 0;

  return <Card title="Read Explorer" hint="Every read with its nearest reference and decision. Filter, then page through; click a row to mark it.">
    <div className="row" style={{ marginBottom: 14, gap: 14, alignItems: "flex-end" }}>
      <div className="seg" role="group" aria-label="Decision filter">
        <button className={decision === "" ? "on" : ""} onClick={() => setDecision("")}>All</button>
        {present.map((d) => <button key={d} className={decision === d ? "on" : ""} onClick={() => setDecision(d)}>{decLabel(d)}</button>)}
      </div>
      <label className="field">Min Identity %
        <input className="input mono" type="number" min="0" max="100" step="0.5" value={minId} placeholder="0"
          onChange={(e) => setMinId(e.target.value)} style={{ width: 110 }} /></label>
      <label className="field" style={{ flex: "1 1 220px" }}>Organism Or Taxonomy
        <span className="search">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true"><circle cx="11" cy="11" r="6.5" /><path d="m20 20-4.2-4.2" /></svg>
          <input className="input" type="search" value={taxon} onChange={(e) => setTaxon(e.target.value)} placeholder="Search, e.g. Thunnus" /></span></label>
    </div>
    {res.error ? <Notice kind="err">{res.error}</Notice> : <div className="table-wrap" style={{ opacity: res.loading ? .55 : 1, transition: "opacity .15s" }}>
      <table className="t">
        <thead><tr><th>Read</th><th>Nearest Organism</th><th>Reference</th><th className="num">Identity</th><th className="num">Coverage</th><th>Decision</th></tr></thead>
        <tbody key={q} className="tbody-anim">
          {res.data && res.data.reads.map((r, k) => { const rowKey = r.Read_ID + ":" + (page * per + k);
            return <tr key={rowKey} className={"click" + (selected === rowKey ? " sel" : "")} aria-selected={selected === rowKey}
              onClick={() => setSelected(selected === rowKey ? null : rowKey)}>
            <td className="mono" style={{ fontSize: 12.5, color: "var(--ink-2)" }}>{r.Read_ID}</td>
            <td title={r.Taxonomy || undefined}>{r.Nearest_Organism ? <span className="org">{r.Nearest_Organism}</span> : <span className="muted">—</span>}</td>
            <td className="mono" style={{ fontSize: 12, color: "var(--ink-muted)" }}>{r.Nearest_Reference_ID || "—"}</td>
            <td className="num">{r.identity_pct === "" ? "—" : <span className="idbar">
              <span className="track"><span className="fill" style={{ width: `${Math.max(0, Math.min(100, Number(r.identity_pct)))}%`, background: decColor(r.Confidence_Decision) }} /></span>
              {fmtPct(r.identity_pct, 2)}</span>}</td>
            <td className="num">{r.coverage_pct === "" ? "—" : fmtPct(r.coverage_pct, 1)}</td>
            <td><Dec d={r.Confidence_Decision} /></td>
          </tr>; })}
          {res.data && !res.data.reads.length && <tr><td colSpan="6" className="muted" style={{ textAlign: "center", padding: 28 }}>No reads match these filters.</td></tr>}
        </tbody>
      </table>
    </div>}
    <div className="row" style={{ justifyContent: "space-between", marginTop: 12 }}>
      <span className="muted mono" style={{ fontSize: 12.5 }}>{total ? `${fmtInt(page * per + 1)}–${fmtInt(Math.min(total, (page + 1) * per))} of ${fmtInt(total)}` : "0 reads"}</span>
      <div className="row">
        <button className="btn sm" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
        <button className="btn sm" disabled={(page + 1) * per >= total} onClick={() => setPage(page + 1)}>Next</button>
      </div>
    </div>
  </Card>;
}

/* ========================================================== compare === */

function CompareView({ ids }) {
  const res = useApi(ids ? `/analytics/compare?run_ids=${encodeURIComponent(ids)}` : null);
  if (!ids) return <Notice>Pick runs to compare on the <a href="#runs">Runs</a> tab.</Notice>;
  if (res.loading) return <div className="stack"><Skel h={70} /><Skel h={260} /></div>;
  if (res.error) return <Notice kind="err">{res.error}</Notice>;
  const d = res.data;
  const runColor = (i) => SERIES[i % SERIES.length];
  const name = (r) => r.filename;
  const maxCell = Math.max(1, ...d.taxa_matrix.flatMap((row) => d.runs.map((r) => row[r.run_id])));
  const decisionKeys = Object.keys(DECISIONS).filter((k) => d.runs.some((r) => r[`pct_${k}`]));

  return <div className="stack">
    <div className="page-head"><div><div className="eyebrow"><a href="#runs" style={{ color: "inherit" }}>Runs</a> / Compare</div><h1>Comparing {d.runs.length} Runs</h1></div></div>
    {d.note && <Notice kind="warn">{d.note}</Notice>}

    <Card title="Side by side">
      <div className="table-wrap" style={{ maxHeight: "none" }}>
        <table className="t">
          <thead><tr><th>Run</th><th>Marker</th><th className="num">Reads</th><th className="num">Assigned</th><th className="num">Median identity</th><th className="num">Taxa</th><th className="num">Top-10 ref share</th></tr></thead>
          <tbody>{d.runs.map((r, i) => <tr key={r.run_id} className="click" onClick={() => go(`run/${r.run_id}`)}>
            <td><span className="dec"><i style={{ background: runColor(i) }} /><b style={{ fontWeight: 500 }}>{name(r)}</b></span><div className="muted mono" style={{ fontSize: 11.5 }}>{r.run_id}</div></td>
            <td style={{ fontSize: 13 }}>{humanize(r.marker)}</td>
            <td className="num">{fmtInt(r.n_reads)}</td><td className="num">{fmtPct(r.assignment_rate_pct)}</td>
            <td className="num">{fmtPct(r.median_identity_assigned)}</td><td className="num">{fmtInt(r.n_distinct_taxa)}</td>
            <td className="num">{fmtPct(r.top10_ref_share_pct)}</td>
          </tr>)}</tbody>
        </table>
      </div>
    </Card>

    <Card title="Decision mix" hint="Share of each run's reads per confidence tier.">
      <div className="stack" style={{ gap: 14 }}>
        <Stacked rows={d.runs.map((r) => ({ label: `${name(r)} · ${humanize(r.marker)}`,
          segments: decisionKeys.map((k) => ({ key: k, label: decLabel(k), value: r[`pct_${k}`] || 0, color: decColor(k) })) }))} />
        <Legend items={decisionKeys.map((k) => ({ label: decLabel(k), color: decColor(k) }))} />
      </div>
    </Card>

    <div className="grid cols-2">
      <Card title="Taxa across runs" hint="Assigned reads per taxon. Darker cells hold more reads.">
        <div className="table-wrap">
          <table className="t">
            <thead><tr><th>Taxon</th>{d.runs.map((r, i) => <th key={r.run_id} className="num"><span className="dec"><i style={{ background: runColor(i) }} />{r.run_id.slice(0, 6)}</span></th>)}</tr></thead>
            <tbody>{d.taxa_matrix.map((row) => <tr key={row.taxon}>
              <td style={{ fontStyle: "italic", fontSize: 13 }}>{row.taxon}</td>
              {d.runs.map((r) => { const v = row[r.run_id], a = v / maxCell;
                return <td key={r.run_id} className="num" style={{ background: v ? `color-mix(in srgb, var(--s1) ${Math.round(8 + a * 55)}%, transparent)` : undefined, color: a > .6 ? "#fff" : undefined }}>{v ? fmtInt(v) : <span className="muted">·</span>}</td>; })}
            </tr>)}</tbody>
          </table>
        </div>
      </Card>
      <Card title="Overlap" hint="Distinct assigned taxa shared between each pair of runs.">
        <table className="t">
          <thead><tr><th>Pair</th><th className="num">Shared</th><th className="num">Only first</th><th className="num">Only second</th><th className="num">Jaccard</th></tr></thead>
          <tbody>{d.taxa_overlap.map((o) => <tr key={o.run_a + o.run_b}>
            <td className="mono" style={{ fontSize: 12.5 }}>{o.run_a.slice(0, 6)} ↔ {o.run_b.slice(0, 6)}</td>
            <td className="num">{o.shared_taxa}</td><td className="num">{o.only_a}</td><td className="num">{o.only_b}</td><td className="num">{fmtNum(o.jaccard, 2)}</td>
          </tr>)}</tbody>
        </table>
      </Card>
    </div>
  </div>;
}

/* ============================================================== chat === */

function renderMarkdown(text) {
  // Models write "~93%" for "about 93%"; unescaped, two of those become strikethrough.
  const html = window.marked ? marked.parse(text.replace(/~/g, "\\~"), { breaks: true }) : text;
  return { __html: window.DOMPurify ? DOMPurify.sanitize(html) : html };   // model output: always sanitise
}

const SUGGEST = {
  run: ["Summarise what this run found and how confident I should be.",
        "Which taxa were detected with confidence?",
        "Are there signs of hubness in this run?",
        "Why did so many reads end up as best guesses?"],
  general: ["What have I uploaded so far, and what did each run find?",
            "Does fine-tuning DNABERT-S improve the results?",
            "Why does the pipeline use BM25 before DNABERT?",
            "How is the reference set for my reads chosen?"],
};

/* Copies text and confirms it in place for a moment. */
function CopyButton({ text }) {
  const [state, setState] = useState("idle");
  async function copy() {
    try { await navigator.clipboard.writeText(text); setState("ok"); }
    catch (_) { setState("fail"); }
    setTimeout(() => setState("idle"), 1600);
  }
  return <button onClick={copy} aria-label={state === "ok" ? "Copied" : "Copy answer"} title={state === "fail" ? "Copy failed" : "Copy"}
    style={{ border: 0, background: "none", color: state === "ok" ? "var(--good-ink)" : "inherit", cursor: "pointer", padding: 0, display: "inline-flex", alignItems: "center", gap: 4 }}>
    {state === "ok" ? <><I.check size={13} /> Copied</> : <I.copy size={13} />}</button>;
}

function ChatDrawer({ open, onClose, runId, setRunId }) {
  const status = useApi(open ? "/chat/status" : null);
  const [sessionId, setSessionId] = useState(null);
  const [sessionRun, setSessionRun] = useState(null);
  const [msgs, setMsgs] = useState([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(null);     // {since}
  const [err, setErr] = useState(null);
  const [now, setNow] = useState(Date.now());
  const bodyRef = useRef(null);
  const taRef = useRef(null);

  // Opening the chat on a different run starts a fresh conversation about it.
  useEffect(() => {
    if (open && runId !== sessionRun && sessionId) { setSessionId(null); setMsgs([]); setErr(null); }
  }, [open, runId]);
  useEffect(() => { if (open && taRef.current) setTimeout(() => taRef.current && taRef.current.focus(), 250); }, [open]);
  useEffect(() => { if (bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight; }, [msgs, pending, err]);
  useEffect(() => { if (!pending) return; const t = setInterval(() => setNow(Date.now()), 500); return () => clearInterval(t); }, [pending]);

  async function send(text) {
    text = (text || "").trim();
    if (!text || pending) return;
    setErr(null); setDraft("");
    setMsgs((m) => [...m, { role: "user", text }]);
    setPending({ since: Date.now() });
    try {
      let sid = sessionId;
      if (!sid) {
        const s = await postJSON("/chat/sessions", runId ? { run_id: runId } : {});
        sid = s.id; setSessionId(sid); setSessionRun(runId);
      }
      const r = await postJSON(`/chat/sessions/${sid}/messages`, { message: text });
      setMsgs((m) => [...m, { role: "assistant", text: r.reply, tools: r.tools_used, model: r.model }]);
    } catch (e) {
      setErr({ message: e.message, retry: text });
      setMsgs((m) => m.slice(0, -1));
    } finally { setPending(null); }
  }

  const newChat = () => { setSessionId(null); setMsgs([]); setErr(null); };
  const st = status.data;
  const elapsed = pending ? Math.round((now - pending.since) / 1000) : 0;

  return <aside className={`ax chat ${open ? "open" : ""}`} aria-label="Analysis assistant" aria-hidden={!open}>
    <div className="chat-head">
      <div className="chat-avatar"><I.spark size={18} /></div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontWeight: 600 }}>Analysis assistant</div>
        <div className="muted mono" style={{ fontSize: 11.5 }}>
          {st ? <>{st.provider} · {st.model}{!st.configured && <span style={{ color: "var(--critical-ink)" }}> · not configured</span>}</> : "…"}</div>
        {runId ? <span className="context-chip">about run {runId}
          <button onClick={() => { setRunId(null); newChat(); }} aria-label="Stop chatting about this run" style={{ border: 0, background: "none", color: "inherit", cursor: "pointer", padding: 0, display: "inline-flex" }}><I.close size={12} /></button></span>
          : <span className="context-chip" style={{ background: "var(--surface-2)", color: "var(--ink-2)" }}>whole project</span>}
      </div>
      <button className="icon-btn" onClick={newChat} aria-label="New conversation" title="New conversation"><I.plus size={16} /></button>
      <button className="icon-btn" onClick={onClose} aria-label="Close assistant"><I.close size={16} /></button>
    </div>

    <div className="chat-body" ref={bodyRef} aria-live="polite">
      {!msgs.length && !pending && <div className="stack" style={{ gap: 14 }}>
        <p className="muted" style={{ margin: 0, fontSize: 13.5 }}>Ask about your uploads, the benchmark results, or the method.
          Answers come from this project's documentation and live result files — it won't answer unrelated questions.</p>
        <div className="suggest">{SUGGEST[runId ? "run" : "general"].map((q) => <button key={q} onClick={() => send(q)}>{q}</button>)}</div>
      </div>}

      {msgs.map((m, i) => m.role === "user"
        ? <div key={i} className="msg user">{m.text}</div>
        : <div key={i} style={{ alignSelf: "flex-start", maxWidth: "92%" }}>
            <div className="msg assistant" style={{ maxWidth: "100%" }} dangerouslySetInnerHTML={renderMarkdown(m.text)} />
            <div className="msg-meta">
              {m.tools && m.tools.map((t, j) => <span key={j} className="tool-chip" title={JSON.stringify(t.input)}>{humanize(t.tool)}{t.error ? " ✕" : ""}</span>)}
              <span>{m.model}</span>
              <CopyButton text={m.text} />
            </div>
          </div>)}

      {pending && <div className="msg assistant" style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span className="typing"><i /><i /><i /></span>
        <span className="muted" style={{ fontSize: 13 }}>{elapsed < 8 ? "Thinking…" : elapsed < 40 ? `Reading results… ${elapsed}s` : `Still working — the model is busy, ${elapsed}s`}</span>
      </div>}

      {err && <div className="msg err"><Notice kind="err"><b>No answer.</b> {err.message}
        <div style={{ marginTop: 8 }}><button className="btn sm" onClick={() => send(err.retry)}>Retry</button></div></Notice></div>}
    </div>

    <div className="chat-foot">
      <form className="composer" onSubmit={(e) => { e.preventDefault(); send(draft); }}>
        <label htmlFor="chat-input" className="sr-only">Message</label>
        <textarea id="chat-input" ref={taRef} rows="1" value={draft} placeholder={runId ? "Ask about this run…" : "Ask about your analyses…"}
          onChange={(e) => { setDraft(e.target.value); e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; }}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(draft); } }} />
        <button className="send" type="submit" disabled={!draft.trim() || !!pending} aria-label="Send"><I.send size={17} /></button>
      </form>
      <div className="muted" style={{ fontSize: 11.5, marginTop: 6, textAlign: "center" }}>Enter to send · Shift+Enter for a new line · answers can take up to a minute</div>
    </div>
  </aside>;
}

/* ============================================================ export === */

window.EdnaAnalysis = { AnalyzeView, RunsView, RunView, CompareView, ChatDrawer };
