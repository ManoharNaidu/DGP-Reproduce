# Prompt layout check, Exp 3: structured zero-shot prompts vs the original-layout prompts

Every field of the original prompt (target review, Key signals line, metapath lines) must appear verbatim in the matching tag of the zero-shot prompt, for every target.

| arm | reference | targets | identical fields | mismatches | mean chars original / zero-shot |
|---|---|---|---|---|---|
| zs_GS0 | gs_G0 | 10,023 | 10,023 | 0 | 1972 / 2654 |
| zs_GS1 | gs_G1 | 10,023 | 10,023 | 0 | 2141 / 2840 |
| zs_GS1b | gs_G1b | 10,023 | 10,023 | 0 | 2123 / 2822 |
| zs_GS2 | gs_G2 | 10,023 | 10,023 | 0 | 2141 / 2840 |
| zs_GS2b | gs_G2b | 10,023 | 10,023 | 0 | 2123 / 2822 |

Result: **PASS**
