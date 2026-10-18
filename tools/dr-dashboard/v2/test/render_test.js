// Headless render test: loads index.html from a running dashboard server in jsdom and prints each section's text.
//   node test/render_test.js http://127.0.0.1:18783/
const {JSDOM, VirtualConsole} = require("jsdom");
const base = process.argv[2] || "http://127.0.0.1:18783/";
(async () => {
  const html = await (await fetch(base)).text();
  const errors = [];
  const vc = new VirtualConsole();
  vc.on("jsdomError", e => errors.push("jsdomError: " + e.message));
  vc.on("error", (...a) => errors.push("console.error: " + a.map(String).join(" ")));
  const dom = new JSDOM(html, {url: base, runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc,
    beforeParse(w) { w.fetch = (u, o) => fetch(new URL(u, base), o); }});
  await new Promise(r => setTimeout(r, 2500));
  const d = dom.window.document, txt = id => (d.getElementById(id) || {textContent: "(missing)"}).textContent.replace(/\s+/g, " ").trim();
  for (const id of ["refresh", "kpis", "overallpct", "stale", "ticker", "gen", "samples", "matrix", "failed", "res", "diag", "gpu"])
    console.log(`--- ${id}: ${txt(id).slice(0, 700)}`);
  console.log("render time:", d.getElementById("refresh").title);
  console.log("ERRORS:", errors.length ? errors : "none");
  process.exit(0);
})();
