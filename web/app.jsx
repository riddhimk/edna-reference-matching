/* eDNA-TaxAI results dashboard.
   React 18 via CDN, transpiled in-browser by Babel standalone -- this machine
   has no Node toolchain, so there is no build step. Charts are hand-rolled SVG
   following the project's data-viz mark specs (thin marks, rounded data-ends,
   recessive grid, hover tooltip, legend whenever two or more series appear). */

const { useState, useEffect, useMemo, useCallback, useRef } = React;

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
    <div className="value" style={accent ? { color: cssVar(accent) } : null}>{value}</div>
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

/* Shared hover tooltip, positioned in viewport space. */
function useTooltip() {
  const [tip, setTip] = useState(null);
  const show = useCallback((e, content) => {
    setTip({ x: e.clientX, y: e.clientY, content });
  }, []);
  const hide = useCallback(() => setTip(null), []);
  const node = tip ? <div className="tooltip" style={{ left: tip.x, top: tip.y }}>{tip.content}</div> : null;
  return { show, hide, node };
}

const TipRows = ({ title, rows }) => <div>
  <div style={{ fontWeight: 650, marginBottom: 4 }}>{title}</div>
  {rows.map(([k, v]) => <div className="t-row" key={k}>
    <span className="t-key">{k}</span><span className="t-val">{v}</span></div>)}
</div>;

/* -------------------------------------------------------------- charts --- */

/* Grouped bars. Data: [{group, values:[{label,value,color}]}] */
function GroupedBars({ data, height = 260, domain, valueFmt = (v) => fmt(v, 4), axisLabel }) {
  const { show, hide, node } = useTooltip();
  const padL = 54, padR = 12, padT = 12, padB = 42;
  const W = 720, H = height;
  const iw = W - padL - padR, ih = H - padT - padB;
  const max = domain ? domain[1] : Math.max(...data.flatMap((g) => g.values.map((v) => v.value))) * 1.12;
  const min = domain ? domain[0] : 0;
  const y = (v) => padT + ih - ((v - min) / (max - min)) * ih;
  const gw = iw / data.length;
  const ticks = 5;

  return <div className="scroll-x">
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", minWidth: 480, display: "block" }} role="img">
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
            return <rect key={v.label} x={bx} y={by} width={bw} height={bh} rx="4" ry="4"
              fill={cssVar(v.color)}
              onMouseMove={(e) => show(e, <TipRows title={`${g.group} · ${v.label}`} rows={[["value", valueFmt(v.value)]]} />)}
              onMouseLeave={hide} style={{ cursor: "pointer" }} />;
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
  const rowH = 30, padL = 224, padR = 60, padT = 6;
  const W = 720, H = height || items.length * rowH + padT * 2;
  const iw = W - padL - padR;
  const max = Math.max(...items.map((i) => i.value)) * 1.05 || 1;
  // Labels are right-anchored at padL, so they must fit the gutter or they
  // overflow the viewBox and get clipped at the left edge.
  const MAXC = Math.floor((padL - 14) / 7.2);
  const clip = (s) => (s.length > MAXC ? s.slice(0, MAXC - 1) + "…" : s);

  return <div className="scroll-x">
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", minWidth: 460, display: "block" }} role="img">
      {items.map((it, i) => {
        const yy = padT + i * rowH, bh = 15, w = Math.max(2, (it.value / max) * iw);
        return <g key={it.label}>
          <text x={padL - 10} y={yy + bh - 2} textAnchor="end" fontSize="12.5" fill="var(--ink-2)">
            {clip(it.label)}</text>
          <rect x={padL} y={yy} width={w} height={bh} rx="4" ry="4" fill={cssVar(it.color || "--s1")}
            onMouseMove={(e) => show(e, <TipRows title={it.label} rows={[["value", valueFmt(it.value)], ...(it.extra || [])]} />)}
            onMouseLeave={hide} style={{ cursor: "pointer" }} />
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
  const padL = 56, padR = 18, padT = 14, padB = 40;
  const W = 720, H = height, iw = W - padL - padR, ih = H - padT - padB;
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

  return <div className="scroll-x">
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", minWidth: 460, display: "block" }} role="img">
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
        <path d={s.points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.y)}`).join(" ")}
          fill="none" stroke={cssVar(s.color)} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        {s.points.map((p, i) => <circle key={i} cx={x(i)} cy={y(p.y)} r="4.5"
          fill={cssVar(s.color)} stroke="var(--surface)" strokeWidth="2"
          onMouseMove={(e) => show(e, <TipRows title={s.label} rows={[[xLabels ? xLabels[i] : `x ${p.x}`, yFmt(p.y)]]} />)}
          onMouseLeave={hide} style={{ cursor: "pointer" }} />)}
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
  let acc = 0;
  return <div>
    <svg viewBox="0 0 720 34" style={{ width: "100%", display: "block" }} role="img">
      {keys.map((k) => {
        const w = (tiers[k] / total) * 720, xx = acc; acc += w;
        return <rect key={k} x={xx + (xx ? 1 : 0)} y="4" width={Math.max(0, w - 2)} height="22" rx="4" ry="4"
          fill={cssVar(colorOf(k))}
          onMouseMove={(e) => show(e, <TipRows title={titleCase(k)}
            rows={[["reads", int(tiers[k])], ["share", ((tiers[k] / total) * 100).toFixed(1) + "%"]]} />)}
          onMouseLeave={hide} style={{ cursor: "pointer" }} />;
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

function Overview({ go }) {
  const prf = useApi("/metrics/prf/comparison?n=5000");
  const health = useApi("/health");
  if (prf.loading || health.loading) return <Loading rows={4} />;
  if (prf.error) return <ErrorBox error={prf.error} />;

  const ks = prf.data.kingdoms;
  const totalFrags = ks.length * 5000;
  const netCalls = ks.reduce((a, k) => a + k.delta.correct_calls, 0);
  const best = ks.reduce((a, b) => (b.frozen.f1 > a.frozen.f1 ? b : a));

  return <div>
    <div className="grid k4" style={{ marginBottom: 18 }}>
      <Tile label="Kingdoms evaluated" value={ks.length} foot="fish · bacteria · eukaryota" />
      <Tile label="Held-out fragments" value={int(totalFrags)} foot="across all kingdoms, n=5,000 each" />
      <Tile label="Net effect of fine-tuning" value={netCalls >= 0 ? `+${netCalls}` : netCalls}
        foot="correct calls gained, of 15,000" accent={Math.abs(netCalls) <= 5 ? "--ink-muted" : "--s1"} />
      <Tile label="Best F1" value={fmt(best.frozen.f1, 3)} foot={titleCase(best.kingdom)} accent="--s3" />
    </div>

    <Card title="The headline finding"
      hint="Fine-tuning the embedding model changes the production pipeline by almost nothing.">
      <div className="callout" style={{ marginBottom: 18 }}>
        Fine-tuning improved the embedding models by up to <strong>+62.1 percentage points</strong> when
        measured in isolation. Inside the full pipeline it is worth <strong>{netCalls >= 0 ? `+${netCalls}` : netCalls} correct
        calls out of {int(totalFrags)}</strong>. BM25 already retrieves nearly every answer the system can
        find, so a better embedding model can only reorder candidates it was already given.
      </div>
      <Legend items={[{ label: "Frozen", color: "--s1" }, { label: "Fine-tuned", color: "--s2" }]} />
      <GroupedBars
        domain={[0.8, 1.0]}
        axisLabel="F1 score"
        valueFmt={(v) => v.toFixed(2)}
        data={ks.map((k) => ({ group: titleCase(k.kingdom).replace(" Large", " (large)"),
          values: [{ label: "Frozen", value: k.frozen.f1, color: "--s1" },
                   { label: "Fine-tuned", value: k.finetuned.f1, color: "--s2" }] }))} />
    </Card>

    <Card title="Per-kingdom detail" hint="At n = 5,000 held-out fragments.">
      <div className="scroll-x"><table>
        <thead><tr><th>Kingdom</th><th>Precision</th><th>Recall</th><th>F1 frozen</th><th>F1 fine-tuned</th>
          <th>Δ calls</th><th>Samples</th></tr></thead>
        <tbody>{ks.map((k) => <tr key={k.kingdom}>
          <td>{titleCase(k.kingdom)}</td>
          <td>{fmt(k.finetuned.precision)}</td>
          <td>{fmt(k.finetuned.recall)}</td>
          <td>{fmt(k.frozen.f1)}</td>
          <td>{fmt(k.finetuned.f1)}</td>
          <td style={{ color: k.delta.correct_calls > 0 ? "var(--good-ink)" : k.delta.correct_calls < 0 ? "var(--critical)" : "var(--ink-muted)" }}>
            {k.delta.correct_calls > 0 ? "+" : ""}{k.delta.correct_calls}</td>
          <td>{k.fragments_per_reference <= 1.05
            ? <span className="pill good">independent</span>
            : <span className="pill warn">{k.fragments_per_reference.toFixed(2)}/ref</span>}</td>
        </tr>)}</tbody>
      </table></div>
      <p className="hint" style={{ marginTop: 14, marginBottom: 0 }}>
        “Samples” flags whether fragments are statistically independent. Only eukaryota (large) draws
        exactly one fragment per held-out reference; the others re-sample, so their numbers carry
        correlated samples.
      </p>
    </Card>

    <div className="grid k2" style={{ marginTop: 18 }}>
      <Card title="Explore the metrics" hint="Precision, recall and F1 at every test-set size.">
        <button className="btn" onClick={() => go("metrics")}>Open metrics →</button></Card>
      <Card title="Explore real reads" hint="Per-read classifications from real seawater samples.">
        <button className="btn" onClick={() => go("reads")}>Open read explorer →</button></Card>
    </div>
  </div>;
}

function Metrics() {
  const [n, setN] = useState(5000);
  const cmp = useApi(`/metrics/prf/comparison?n=${n}`, [n]);
  const all = useApi("/metrics/prf");
  if (cmp.loading || all.loading) return <Loading rows={4} />;
  if (cmp.error) return <ErrorBox error={cmp.error} />;

  const ks = cmp.data.kingdoms;
  return <div>
    <div className="filters">
      <label>Test-set size</label>
      {[1000, 2000, 5000].map((v) =>
        <button key={v} className="btn" onClick={() => setN(v)}
          style={n === v ? { background: "var(--surface-2)", fontWeight: 650, borderColor: "var(--s1)" } : null}>
          {int(v)}</button>)}
    </div>

    <Card title="Precision, recall and F1" hint={`Production pipeline scored as a decision system, n = ${int(n)} fragments.`}>
      <Legend items={[{ label: "Precision", color: "--s1" }, { label: "Recall", color: "--s2" }, { label: "F1", color: "--s3" }]} />
      <GroupedBars domain={[0.8, 1.0]} valueFmt={(v) => v.toFixed(2)} axisLabel="score"
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
          <tr key={k.kingdom + "f"}><td rowSpan="2">{titleCase(k.kingdom)}</td><td>frozen</td>
            <td>{fmt(k.frozen.precision)}</td><td>{fmt(k.frozen.recall)}</td><td>{fmt(k.frozen.f1)}</td>
            <td>{int(k.frozen.tp)}</td><td rowSpan="2">{k.delta.f1 >= 0 ? "+" : ""}{fmt(k.delta.f1, 4)}</td>
            <td rowSpan="2" style={{ color: k.delta.correct_calls > 0 ? "var(--good-ink)" : k.delta.correct_calls < 0 ? "var(--critical)" : "var(--ink-muted)" }}>
              {k.delta.correct_calls > 0 ? "+" : ""}{k.delta.correct_calls}</td></tr>,
          <tr key={k.kingdom + "t"}><td>fine-tuned</td>
            <td>{fmt(k.finetuned.precision)}</td><td>{fmt(k.finetuned.recall)}</td><td>{fmt(k.finetuned.f1)}</td>
            <td>{int(k.finetuned.tp)}</td></tr>])}</tbody>
      </table></div>
    </Card>

    <Card title="Stability across test-set size" hint="Nested prefixes of one 5,000-fragment run, so sizes are directly comparable.">
      <Legend items={all.data.kingdoms.map((k, i) => ({ label: titleCase(k.kingdom), color: SERIES[i] }))} />
      <LineChart yLabel="F1 (fine-tuned)" yFmt={(v) => v.toFixed(3)}
        xLabels={["1,000", "2,000", "5,000"]}
        series={all.data.kingdoms.map((k, i) => ({
          label: titleCase(k.kingdom), color: SERIES[i],
          points: [1000, 2000, 5000].map((sz, j) => ({ x: j,
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
  const [kingdom, setKingdom] = useState("bacteria");
  const arch = useApi(`/metrics/architecture?kingdom=${kingdom}`, [kingdom]);
  if (arch.loading) return <Loading rows={3} />;
  if (arch.error) return <ErrorBox error={arch.error} />;
  const kd = arch.data.kingdoms[0];
  const arms = kd.architectures;
  const label = (a) => a.replace(/_/g, " ").replace("BM25 MMR", "BM25→MMR").replace("RRF", "RRF");

  return <div>
    <div className="filters">
      <label>Kingdom</label>
      {["bacteria", "fish", "eukaryota"].map((k) =>
        <button key={k} className="btn" onClick={() => setKingdom(k)}
          style={kingdom === k ? { background: "var(--surface-2)", fontWeight: 650, borderColor: "var(--s1)" } : null}>
          {titleCase(k)}</button>)}
    </div>

    <div className="grid k3" style={{ marginBottom: 18 }}>
      <Tile label="Held-out fragments" value={int(kd.n_holdout_fragments)} />
      <Tile label="Identity threshold" value={kd.identity_threshold + "%"} foot="bar for a confident call" />
      <Tile label="Architectures compared" value={arms.length} foot="each through the same reranker" />
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
      <p className="hint" style={{ marginTop: 14, marginBottom: 0 }}>
        Lower redundancy means a more diverse candidate set — the one thing fine-tuning reliably improves.
      </p>
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
      hint="Recall@10 on an identical 1,000-fragment held-out split. Higher is better.">
      <HBars valueFmt={(v) => (v * 100).toFixed(1) + "%"}
        items={cmpRows.map((r, i) => ({ label: r.label, value: r.value,
          color: i === 0 ? "--s3" : "--s1", extra: [["recall@1", fmt(r.r1, 3)]] }))} />
      <p className="hint" style={{ marginTop: 10, marginBottom: 0 }}>
        DNABERT-S wins decisively: it is the only candidate trained specifically to tell species apart.
      </p>
    </Card>

    <Card title="Fine-tuning loss curves" hint="Corrected (v2) training runs. Lower is better, but see the note.">
      <Legend items={corrected.map((m, i) => ({ label: m.kingdom, color: SERIES[i % SERIES.length] }))} />
      <LineChart yLabel="mean loss" yFmt={(v) => v.toFixed(2)} xLabels={["Epoch 1", "Epoch 2"]}
        domain={[0, Math.max(...corrected.flatMap((m) => m.mean_loss_per_epoch)) * 1.1]}
        series={corrected.map((m, i) => ({ label: m.kingdom, color: SERIES[i % SERIES.length],
          points: m.mean_loss_per_epoch.map((v, j) => ({ x: j, y: v })) }))} />
      <div className="callout" style={{ marginTop: 14 }}>
        A falling loss is <strong>not</strong> evidence of better retrieval. An earlier bacteria run's loss fell
        steadily while its recall collapsed — the held-out metrics are what settle it.
      </div>
    </Card>

    <Card title="All checkpoints" hint="Every fine-tuning run recorded in this project.">
      <div className="scroll-x"><table>
        <thead><tr><th>Checkpoint</th><th>Kingdom</th><th>Train refs</th><th>Distinct seqs</th>
          <th>Epochs</th><th>Loss 1 → 2</th><th>Minutes</th></tr></thead>
        <tbody>{models.data.models.map((m) => <tr key={m.name}>
          <td className="mono">{m.name.replace("dnaberts_finetuned_", "")}</td>
          <td>{m.kingdom || "—"}</td>
          <td>{m.n_train_references ? int(m.n_train_references) : "—"}</td>
          <td>{m.n_distinct_sequences ? int(m.n_distinct_sequences) : "—"}</td>
          <td>{m.epochs ?? "—"}</td>
          <td>{m.mean_loss_per_epoch ? m.mean_loss_per_epoch.map((v) => v.toFixed(3)).join(" → ") : "—"}</td>
          <td>{m.train_seconds ? (m.train_seconds / 60).toFixed(0) : "—"}</td></tr>)}</tbody>
      </table></div>
      <p className="hint" style={{ marginTop: 14, marginBottom: 0 }}>
        Rows with dashes are superseded v1 runs, trained before the corrected recipe; their metadata
        predates several of these fields.
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
          <option key={d.dataset} value={d.dataset}>{titleCase(d.dataset)} · {int(d.n_reads)} reads</option>)}
      </select>
      <label>Decision</label>
      <select value={decision} onChange={(e) => { setDecision(e.target.value); setPage(0); }}>
        <option value="">all</option>
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
          foot={summary.data.score_column} />
        <Tile label="Distinct organisms" value={int(summary.data.distinct_values)}
          foot={"grouped by " + summary.data.grouped_by} />
      </div>

      <Card title="Confidence composition" hint="How the pipeline's decisions break down across this dataset.">
        <StackedTiers tiers={summary.data.decisions} total={summary.data.n_reads} />
      </Card>

      <div className="grid k2" style={{ marginTop: 18 }}>
        <Card title="Score distribution" hint={`Percentiles of ${summary.data.score_column}.`}>
          <HBars valueFmt={(v) => v.toFixed(2)} items={[
            ["min", summary.data.score_distribution.min], ["p25", summary.data.score_distribution.p25],
            ["median", summary.data.score_distribution.median], ["p75", summary.data.score_distribution.p75],
            ["max", summary.data.score_distribution.max]]
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
    <Card title="Per-read result sets" hint="Every classification run stored in this project, newest experiments first.">
      <div className="scroll-x"><table>
        <thead><tr><th>Dataset</th><th>Kingdom</th><th>Part</th><th>Accession</th><th>Reads</th><th>Score column</th></tr></thead>
        <tbody>{list.data.datasets.map((d) => <tr key={d.dataset}>
          <td><div style={{ fontWeight: 600 }}>{titleCase(d.dataset)}</div>
            <div style={{ fontSize: 12, color: "var(--ink-muted)", maxWidth: 520, whiteSpace: "normal" }}>{d.description}</div></td>
          <td>{titleCase(d.kingdom)}</td><td>{d.part}</td><td className="mono">{d.accession}</td>
          <td>{int(d.n_reads)}</td><td className="mono">{d.score_column || "—"}</td></tr>)}</tbody>
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

const PAGES = {
  overview:     { title: "Overview",            sub: "Headline results across all three kingdoms",        C: Overview },
  metrics:      { title: "Precision / Recall",  sub: "The pipeline scored as a decision system",          C: Metrics },
  architecture: { title: "Architecture",        sub: "Seven retrieval architectures, head to head",       C: Architecture },
  models:       { title: "Models",              sub: "Fine-tuning runs and embedding-model comparison",   C: Models },
  reads:        { title: "Read Explorer",       sub: "Per-read classifications from real seawater",       C: Reads },
  datasets:     { title: "Datasets",            sub: "What data exists and where it came from",           C: Datasets },
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
    return matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("edna-theme", theme); } catch (_) {}
  }, [theme]);
  return [theme, setTheme];
}

function App() {
  const [page, setPage] = useState(() => (location.hash || "#overview").slice(1));
  const [theme, setTheme] = useTheme();
  useEffect(() => {
    const on = () => setPage((location.hash || "#overview").slice(1));
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  const go = (p) => { location.hash = "#" + p; };
  const cur = PAGES[page] || PAGES.overview;
  const View = cur.C;

  return <div className="shell">
    <nav className="sidebar">
      <div className="brand">
        <h1>eDNA-TaxAI</h1>
        <p>Results dashboard</p>
      </div>
      {Object.entries(PAGES).map(([k, v]) =>
        <button key={k} className={"navlink" + (k === page ? " active" : "")} onClick={() => go(k)}>
          <span className="dot" />{v.title}</button>)}
      <div className="sidebar-foot">
        <button className="navlink" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
          <span className="dot" />{theme === "dark" ? "Light mode" : "Dark mode"}</button>
        <a className="navlink" href="/docs" target="_blank" rel="noreferrer"><span className="dot" />API docs ↗</a>
      </div>
    </nav>
    <div className="main">
      <header className="topbar">
        <div><h2>{cur.title}</h2><div className="sub">{cur.sub}</div></div>
        <span className="pill">read-only · no inference</span>
      </header>
      <main className="content">
        <Boundary viewKey={page}><View go={go} /></Boundary>
      </main>
    </div>
  </div>;
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
