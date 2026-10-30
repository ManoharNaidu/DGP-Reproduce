# Prompt layout check, Exp 2: structured zero-shot prompts vs the original-layout prompts

Every field of the original prompt (target review, Key signals line, metapath lines) must appear verbatim in the matching tag of the zero-shot prompt, for every target.

| arm | reference | targets | identical fields | mismatches | mean chars original / zero-shot |
|---|---|---|---|---|---|
| zs_S0 | dr_R0 | 10,023 | 10,023 | 0 | 3163 / 3845 |
| zs_S1 | dr_R1 | 10,023 | 10,023 | 0 | 3356 / 4055 |
| zs_S1b | dr_R1b | 10,023 | 10,023 | 0 | 3331 / 4030 |
| zs_S2 | dr_R2 | 10,023 | 10,023 | 0 | 3356 / 4055 |
| zs_S2b | dr_R2b | 10,023 | 10,023 | 0 | 3331 / 4030 |

Result: **PASS**
