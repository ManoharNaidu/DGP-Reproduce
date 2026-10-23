// Headless UI test for the traceability additions: header provenance, manifest link, gold winner row, box plots,
// failed-run error blocks, Probe AUROC help tip, config-diff and samples modals (open, content, Escape closes).
//   node test/trace_ui_test.js http://127.0.0.1:18795/            # exits 1 on any failed check or console error
const {JSDOM, VirtualConsole} = require("jsdom");
const base = process.argv[2] || "http://127.0.0.1:18795/";
const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  const html = await (await fetch(base)).text();
  const errors = [], fails = [];
  const vc = new VirtualConsole();
  vc.on("jsdomError", e => errors.push("jsdomError: " + e.message));
  vc.on("error", (...a) => errors.push("console.error: " + a.map(String).join(" ")));
  const dom = new JSDOM(html, {url: base, runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc,
    beforeParse(w) { w.fetch = (u, o) => fetch(new URL(u, base), o); }});
  await sleep(2500);
  const w = dom.window, d = w.document, $ = id => d.getElementById(id);
  const txt = n => (n ? n.textContent : "(missing)").replace(/\s+/g, " ").trim();
  const check = (name, ok, detail) => { console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? " · " + detail : ""}`); if (!ok) fails.push(name); };
  const state = await (await fetch(new URL("api/state", base))).json();
  const study = d.querySelector("#prov b") ? txt($("prov")) : "";

  check("provenance line", /git /.test(txt($("prov"))) && /branch /.test(txt($("prov"))), txt($("prov")).slice(0, 160));
  check("hypothesis line", /^Hypothesis:/.test(txt($("hyp"))), txt($("hyp")).slice(0, 120));
  check("manifest link", /api\/manifest\?study=(fa|dr)&download=1/.test($("manifestdl").getAttribute("href")), $("manifestdl").getAttribute("href"));
  const win = (state.results || {}).winner;
  const goldRows = [...d.querySelectorAll("#res tr.gold")];
  check("dr gold row = server winner", win ? goldRows.length === 1 && txt(goldRows[0]).startsWith((state.results.arms.find(a => a.arm === win.arm) || {}).label || "?") : goldRows.length === 0,
        win ? `winner ${win.arm} · ${txt($("drwin"))}` : "no winner");
  const boxes = d.querySelectorAll("#res td .ms svg").length;
  check("box plots next to mean ± std", boxes > 0 || !(state.results.arms || []).some(a => Object.keys(a.per_seed || {}).length), `${boxes} box plots`);
  const failedCells = [...d.querySelectorAll(".cell.failed")];
  const withErr = failedCells.filter(c => c.querySelector(".err .kind"));
  const failedJobs = (state.queue.jobs || []).filter(j => j.state === "failed" && j.error).length + ((state.fa.queue || {}).jobs || []).filter(j => j.state === "failed" && j.error).length;
  check("failed cells show error kind + summary", withErr.length === failedJobs, `${withErr.length} / ${failedJobs} · ${withErr.map(c => txt(c.querySelector(".err .sum"))).join(" | ").slice(0, 200)}`);
  const scripted = d.querySelectorAll(".err script, #modalroot script").length;
  check("error text is inert text (no injected elements)", scripted === 0);
  const tip = d.querySelector("#diag .help");
  check("Probe AUROC help tip", !tip || tip.getAttribute("data-tip") === "Measures how easily a classifier can extract the label from the generated text alone.",
        tip ? "present" : "diagnostics not rendered on this tree");

  for (const [btn, label] of [["cfgdiff-dr", "config diff dr"], ["cfgdiff-fa", "config diff fa"], ["samples-btn", "samples"]]) {
    $(btn).dispatchEvent(new w.MouseEvent("click", {bubbles: true}));
    await sleep(1200);
    const modal = d.querySelector("#modalroot .modal");
    check(`${label} modal opens`, !!modal, modal ? txt(modal).slice(0, 230) : "");
    if (modal && label.startsWith("config")) {
      const rows = modal.querySelectorAll("table.data tr").length;
      check(`${label} shows only differing fields or unknown`, rows > 1 || /unknown|No differing/.test(txt(modal)), `${Math.max(0, rows - 1)} differing fields`);
    }
    if (modal && label === "samples") {
      const n = modal.querySelectorAll(".sample").length;
      check("samples rendered as text", n > 0 || /No samples|no prompt file/.test(txt(modal)), `${n} samples`);
      check("no test labels shown", ![...modal.querySelectorAll(".sample")].some(s => /split test/.test(txt(s)) && !/hidden/.test(txt(s))));
    }
    d.dispatchEvent(new w.KeyboardEvent("keydown", {key: "Escape", bubbles: true}));
    await sleep(100);
    check(`${label} closes on Escape`, !d.querySelector("#modalroot .modal"));
    check(`${label} focus returns to opener`, d.activeElement === $(btn));
  }
  console.log("ERRORS:", errors.length ? errors : "none");
  process.exit(errors.length || fails.length ? 1 : 0);
})();
