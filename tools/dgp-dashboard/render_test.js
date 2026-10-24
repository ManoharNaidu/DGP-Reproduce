// Headless render test: loads the page from a running server in jsdom, checks every section, the metric switch,
// the modal, and that no section shows NaN / undefined. Exit 1 on any failure or console error.
//   node tools/dgp-dashboard/render_test.js http://127.0.0.1:8799/
// jsdom: tools/dr-dashboard/v2/test/node_modules (already installed) or anywhere on NODE_PATH.
let jsdom;
try { jsdom = require("jsdom"); } catch (e) { jsdom = require(require("path").join(__dirname, "..", "dr-dashboard", "v2", "test", "node_modules", "jsdom")); }
const {JSDOM, VirtualConsole} = jsdom;
const base = process.argv[2] || "http://127.0.0.1:8799/";
const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  const html = await (await fetch(base)).text();
  const errors = [], fails = [];
  const vc = new VirtualConsole();
  vc.on("jsdomError", e => errors.push("jsdomError: " + e.message));
  vc.on("error", (...a) => errors.push("console.error: " + a.map(String).join(" ")));
  const dom = new JSDOM(html, {url: base, runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc,
    beforeParse(w) { w.fetch = (u, o) => fetch(new URL(u, base), o); }});
  await sleep(4000);
  const w = dom.window, d = w.document, $ = id => d.getElementById(id), txt = n => (n ? n.textContent : "").replace(/\s+/g, " ").trim();
  const check = (name, ok, detail) => { console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? " · " + String(detail).slice(0, 220) : ""}`); if (!ok) fails.push(name); };
  const state = await (await fetch(new URL("api/state", base))).json();
  // default view: the current run only
  check("default view is the current run", /Current run/.test(txt($("ov"))) && /FLAG-architecture study/.test(txt($("ex"))) && !/Main DGP reproduction/.test(txt($("ex"))), txt($("ex")).slice(0, 120));
  check("superseded launch named, not mixed in", !(state.overview.batches || []).some(b => b.superseded_by) || /Replaced earlier launch/.test(txt($("ov"))));
  check("current-run events only", (state.overview.events || []).every(e => !state.overview.current_since || e.t >= state.overview.current_since));
  const all = [...d.querySelectorAll("#scope button")].find(b => b.textContent === "All studies");
  all.dispatchEvent(new w.MouseEvent("click", {bubbles: true})); await sleep(300);
  for (const [id, need] of [["ov", /Overall progress/], ["ex", /Main DGP reproduction/], ["rs", /Change vs baseline/], ["in", /Strongest result/], ["if", /Download results CSV/]])
    check(`section ${id} rendered`, need.test(txt($(id))), txt($(id)).slice(0, 160));
  for (const id of ["ov", "ex", "rs", "in", "if"]) check(`section ${id} has no NaN/undefined`, !/\bNaN\b|undefined/.test(txt($(id))));
  check("studies labelled historical / live", /historical/.test(txt($("ex"))) && /live/.test(txt($("ex"))));
  const grids = d.querySelectorAll("#ex .seeds").length, arms = state.studies.reduce((a, s) => a + s.arms.length, 0);
  check("one five-seed grid per configuration", grids === arms && d.querySelectorAll("#ex .seeds .sd").length === arms * 5, `${grids} grids / ${arms} arms`);
  const deltaRows = state.studies.reduce((a, s) => a + s.arms.filter(x => !x.is_baseline).length, 0);
  check("delta chart rows", d.querySelectorAll("#rs svg.chart")[0].querySelectorAll("text.lbl").length === deltaRows, `${deltaRows} arms`);
  const paperLine = [...d.querySelectorAll("#rs svg text")].some(t => t.textContent === "paper");
  check("paper reference drawn", !!state.paper === paperLine);
  const before = txt($("rs"));
  const btn = [...d.querySelectorAll("#metric button")].find(b => b.textContent === "AUPRC");
  btn.dispatchEvent(new w.MouseEvent("click", {bubbles: true})); await sleep(200);
  const now = [...d.querySelectorAll("#metric button")].find(b => b.textContent === "AUPRC");      // buttons are redrawn on click
  check("metric switch redraws results", txt($("rs")) !== before && /AUPRC/.test(txt($("rs"))) && now.getAttribute("aria-pressed") === "true");
  const live = state.studies.find(s => s.key === "fa");
  const liveEmpty = live.arms.every(a => a.n === 0);
  check("live study without runs shows no fabricated numbers", !liveEmpty || /no paired runs yet/.test(txt(d.querySelectorAll("#rs svg.chart")[0])));
  const link = d.querySelector("#ex details a");
  if (link) { link.dispatchEvent(new w.MouseEvent("click", {bubbles: true})); await sleep(1500);
    const m = d.querySelector("#modalroot .modal"); check("run manifest modal", !!m && /"run_id"/.test(txt(m)), m && txt(m).slice(0, 80));
    d.dispatchEvent(new w.KeyboardEvent("keydown", {key: "Escape", bubbles: true})); await sleep(100); check("modal closes on Escape", !d.querySelector("#modalroot .modal")); }
  check("live indicator", /Live|Stale|Offline/.test(txt($("live"))), txt($("live")));
  console.log("ERRORS:", errors.length ? errors : "none");
  process.exit(errors.length || fails.length ? 1 : 0);
})();
