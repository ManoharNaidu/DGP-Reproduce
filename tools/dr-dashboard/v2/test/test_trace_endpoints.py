"""Endpoint tests for the traceability additions (manifest, samples, provenance, error snippets), on mock trees only.

    /venv/main/bin/python -I tools/dr-dashboard/v2/test/test_trace_endpoints.py      # needs numpy (mock labels.npy)

Builds the mock trees in a temporary directory, starts server.py on a free port with --root, and checks that the
server answers from recorded data only and never writes into the data tree (its own state goes to .dashboard_state).
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def snapshot(root):
    out = {}
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x != ".dashboard_state"]
        for f in files:
            p = os.path.join(d, f)
            st = os.stat(p)
            out[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns)
    return out


class Server:
    def __init__(self, root):
        self.root, self.port = root, free_port()
        self.proc = subprocess.Popen([PY, "-I", os.path.join(V2, "server.py"), str(self.port), "--root", root,
                                      "--no-insights"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(50):
            try:
                self.get("/api/state")
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("server did not start")

    def get(self, path, raw=False):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=20) as r:
            body = r.read()
            return (r, body) if raw else json.loads(body)

    def stop(self):
        self.proc.terminate()
        self.proc.wait(10)


class TraceEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = cls.tmp.name
        for args in (["make_mock.py", t], ["make_fa_mock.py", os.path.join(t, "fa")],
                     ["make_trace_mock.py", os.path.join(t, "full"), os.path.join(t, "fa")]):
            subprocess.run([PY, os.path.join(V2, "mock", args[0]), *args[1:]], check=True, stdout=subprocess.DEVNULL)
        cls.dr_root, cls.fa_root = os.path.join(t, "full"), os.path.join(t, "fa")
        cls.before = {r: snapshot(r) for r in (cls.dr_root, cls.fa_root)}
        cls.dr, cls.fa = Server(cls.dr_root), Server(cls.fa_root)

    @classmethod
    def tearDownClass(cls):
        cls.dr.stop()
        cls.fa.stop()
        cls.tmp.cleanup()

    def test_manifest_dr_unknowns_and_winner(self):
        m = self.dr.get("/api/manifest?study=dr")
        self.assertEqual(m["schema"], "dgp-experiment-manifest-v1")
        self.assertEqual(m["run_hypothesis"], "unknown")           # no batch record in the dr mock
        self.assertEqual(m["cli_commands"], "unknown")
        self.assertEqual(m["git"]["branches"], "unknown")
        self.assertTrue(all(c["commit"] == "0123456789abcdef0123456789abcdef01234567" for c in m["git"]["commits"]))
        means = {a["arm"]: a["summary"]["val_auroc"]["mean"] for a in m["arms"] if a["summary"]["val_auroc"]}
        self.assertEqual(m["winner"]["arm"], max(means, key=means.get))
        self.assertEqual((m["winner"]["metric"], m["winner"]["split"]), ("auroc", "val"))
        r2 = next(a for a in m["arms"] if a["arm"] == "R2")
        self.assertEqual(r2["hyperparameters"].get("residual.alpha"), 0.1)
        self.assertEqual(r2["prompt_version"], "v1-mock")
        never = next(a for a in m["arms"] if not a["seeds_finished"] and a["hyperparameters_source"] == "unknown")
        self.assertEqual(never["hyperparameters"], "unknown")

    def test_manifest_fa_batch_record_and_download(self):
        m = self.fa.get("/api/manifest?study=fa")
        self.assertEqual(m["run_hypothesis"], ["Mock hypothesis: FLAG-style D lines help."])
        self.assertEqual(m["git"]["branches"], ["mock-branch"])
        self.assertEqual(m["cli_commands"], ["nohup scripts/fa_queue.sh > logs/flag_arch/queue.log 2>&1 &"])
        r, body = self.fa.get("/api/manifest?study=fa&download=1", raw=True)
        self.assertIn("experiment_manifest_fa.json", r.headers.get("Content-Disposition", ""))
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.fa.get("/api/manifest?study=nope")
        self.assertEqual(e.exception.code, 400)

    def test_manifest_cli_matches_endpoint(self):
        out = subprocess.run([PY, "-I", os.path.join(V2, "server.py"), "--root", self.fa_root, "--manifest", "fa"],
                             capture_output=True, text=True, check=True).stdout
        cli, api = json.loads(out), self.fa.get("/api/manifest?study=fa")
        cli.pop("generated_at")
        api.pop("generated_at")
        self.assertEqual(cli, api)

    def test_samples(self):
        d = self.dr.get("/api/samples?study=dr&arm=R1&n=5")
        self.assertTrue(1 <= len(d["samples"]) <= 5)
        self.assertEqual(d["source"], "datasets/cache/prompts/dr_R1_abc123.jsonl")
        for s in d["samples"]:
            self.assertLessEqual(len(s["prompt"]), 1500)
            self.assertEqual(s["truncated"], s["prompt_chars"] > 1500)
            if s["split"] == "test":
                self.assertEqual(s["label"], "hidden (test split)")
            else:
                self.assertIn(s["label"]["value"], (0, 1))
            self.assertIn("rating", s["features"])
            self.assertEqual(s["features"]["num_neighbors_trimmed"], "48")
        self.assertEqual(len(self.dr.get("/api/samples?study=dr&arm=R1&n=99")["samples"]) <= 5, True)
        none = self.dr.get("/api/samples?study=dr&arm=R2")              # no prompt file for R2 in the mock
        self.assertEqual((none["samples"], none["source"]), ([], "unknown"))
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.dr.get("/api/samples?study=dr&arm=../../etc")
        self.assertEqual(e.exception.code, 400)

    def test_state_provenance_errors_winner(self):
        s = self.fa.get("/api/state")
        p = s["provenance"]["fa"]
        self.assertEqual((p["git_branch"], p["run_hypothesis"]), ("mock-branch", "Mock hypothesis: FLAG-style D lines help."))
        self.assertEqual(s["provenance"]["dr"]["run_hypothesis"], "unknown")
        failed = [j for j in s["fa"]["queue"]["jobs"] if j["state"] == "failed"]
        self.assertEqual([j["job"] for j in failed], ["fa_F0_s1"])
        self.assertEqual(failed[0]["error"]["kind"], "parse error")
        self.assertIn("JSONDecodeError", failed[0]["error"]["summary"])
        d = self.dr.get("/api/state")
        oom = d["queue"]["failed"][0]["error"]
        self.assertEqual(oom["kind"], "OOM")
        self.assertIn("winner", d["results"])
        self.assertIn("val", next(iter(d["results"]["arms"][0]["per_seed"].values())))

    def test_zz_tree_unchanged(self):                      # runs last: every endpoint above was read-only
        for r in (self.dr_root, self.fa_root):
            self.assertEqual(snapshot(r), self.before[r], f"server wrote into {r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
