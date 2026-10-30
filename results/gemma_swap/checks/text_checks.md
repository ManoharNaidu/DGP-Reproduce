# Text caches: Gemma-2-9B vs Qwen3-8B (same prompts, same inputs)

class words = helpful / unhelpful / useful / useless / not helpful / not useful (as in dr_text_diagnostics.py). hit cap = generation stopped at max_new_tokens (4 x budget + 16).

| text | model | lines | empty % | multi-line % | tokens mean / p95 | hit cap % | class words % |
|---|---|---|---|---|---|---|---|
| neighbour summaries (node_summary, B=10) | gemma | 35,747 / 35,747 | 0.0 | 0.02 | 17.7 / 28.0 | 0.09 | 0.11 |
| neighbour summaries (node_summary, B=10) | qwen | 36,261 / 35,747 | 0.0 | 0.22 | 23.7 / 56.0 | 5.11 | 0.08 |
| metapath summaries (B=10) | gemma | 119,829 / 119,829 | 0.0 | 0.01 | 18.3 / 27.0 | 0.01 | 0.04 |
| metapath summaries (B=10) | qwen | 119,829 / 119,829 | 0.0 | 0.2 | 38.5 / 56.0 | 26.95 | 0.09 |
| D line · FLAG prompt (dr_discriminative) | gemma | 10,023 / 10,023 | 0.0 | 0.0 | 28.7 / 37.0 | 0.0 | 74.73 |
| D line · FLAG prompt (dr_discriminative) | qwen | 10,023 / 10,023 | 0.0 | 0.54 | 30.7 / 41.0 | 0.03 | 92.37 |
| D line · no-verdict (dr_discriminative_nv) | gemma | 10,023 / 10,023 | 0.0 | 0.0 | 25.1 / 33.0 | 0.0 | 4.27 |
| D line · no-verdict (dr_discriminative_nv) | qwen | 10,023 / 10,023 | 0.0 | 0.1 | 26.7 / 40.0 | 0.07 | 0.26 |
| R line (dr_residual) | gemma | 10,023 / 10,023 | 0.0 | 0.0 | 14.6 / 20.0 | 0.0 | 0.0 |
| R line (dr_residual) | qwen | 10,023 / 10,023 | 0.0 | 0.72 | 22.6 / 37.0 | 0.24 | 0.01 |

## Samples (same input, both models)

- **neighbour summaries (node_summary, B=10)**, key `31301`
  - Gemma: 'Disappointing "guerilla" Disney film with no plot or story, resembling Eraserhead.'
  - Qwen: "The film's only merit was guerrilla filmmaking. It lacks plot, genre, and coherence. A disappointment for Disney fans. Two stars for the filmmaking technique."
- **neighbour summaries (node_summary, B=10)**, key `27089`
  - Gemma: 'Amazing actors, great show, season 5 was fantastic!'
  - Qwen: 'I love this show. Actors are amazing. Season 5 was great!'
- **neighbour summaries (node_summary, B=10)**, key `5026`
  - Gemma: 'Grandbaby loves and learns from Diego.'
  - Qwen: 'Grandbaby loves Diego, learns well, well done.'
- **metapath summaries (B=10)**, key `31371|RPR-RPR|36dbdf1c82de`
  - Gemma: '"Reign" is an entertaining historical fiction show about Mary, Queen of Scotland.'
  - Qwen: 'Entertaining, fictionalized retelling with historical license.'
- **metapath summaries (B=10)**, key `27164|RPR-RSR|383b29351873`
  - Gemma: 'Hilarious British comedy, nostalgic Cold War setting, great acting.'
  - Qwen: 'BEST show ever. Addicted, no cable. Fun, hits home. Loved Cold War, fought Russia, China. Dawn French humor, British shows, gift of laughter. Great acting, story, action. Suggest for movie night.'
- **metapath summaries (B=10)**, key `4957|RPR-RUR|03c3fd9f7561`
  - Gemma: 'Nostalgic, well-made, but lacking the edge of earlier seasons.'
  - Qwen: 'Strong actors, good writing, and interesting history. Worth watching next season.'
- **D line · FLAG prompt (dr_discriminative)**, key `31377`
  - Gemma: "The review provides a subjective opinion on the series' quality and offers specific suggestions for improvement, making it potentially helpful to other viewers."
  - Qwen: "The review provides specific feedback on the series' acting, storyline, and content, offering constructive criticism and highlighting both strengths and areas for improvement, making it helpful."
- **D line · FLAG prompt (dr_discriminative)**, key `27183`
  - Gemma: "The review provides specific criticisms of the show's portrayal of mental illness, character development, and storytelling techniques, offering both negative and positive aspects, making it potentially helpful to other viewers."
  - Qwen: 'The review is unhelpful because it contains contradictory statements, lacks a clear focus, and includes personal opinions without providing specific examples or constructive feedback about the product.'
- **D line · FLAG prompt (dr_discriminative)**, key `4973`
  - Gemma: "The review provides a detailed and personal account of the reviewer's experience with the show, highlighting its strengths and comparing it to other similar shows, making it helpful."
  - Qwen: "The review is helpful because it provides specific praise for the show's intelligent writing, realistic scenarios, and well-developed characters, which are valuable insights for potential viewers."
- **D line · no-verdict (dr_discriminative_nv)**, key `31377`
  - Gemma: 'The review provides specific details about the acting, storyline, and personal preferences regarding action and nudity.'
  - Qwen: 'The review provides specific feedback on acting, storyline, and content elements like drama, action, nudity, and character portrayal.'
- **D line · no-verdict (dr_discriminative_nv)**, key `27183`
  - Gemma: "The review provides specific examples of plot elements and character portrayals, both positive and negative, and offers detailed feedback on the episode's handling of mental illness."
  - Qwen: 'The review critiques the over-dramatization and stereotypical portrayal of mental illness in the episode, while praising the character development and patient-centric approach.'
- **D line · no-verdict (dr_discriminative_nv)**, key `4973`
  - Gemma: "The review provides detailed insights into the show's writing, acting, and realism, comparing it to other similar shows and highlighting specific aspects that make it stand out."
  - Qwen: "The review discusses the reviewer's long-standing interest in family-centered dramas, their initial hesitation to commit to the series, and their eventual positive experience with the show's intelligent writing, realistic scenarios, and strong acting."
- **R line (dr_residual)**, key `31377`
  - Gemma: 'This review discusses a series about Mary of Scotland.'
  - Qwen: 'A historical drama series focusing on Mary of Scotland with strong acting and a compelling storyline.'
- **R line (dr_residual)**, key `27183`
  - Gemma: 'The reviewer discusses the portrayal of mental illness in a television series.'
  - Qwen: 'The review discusses the portrayal of mental illness in a TV series, highlighting both strengths and criticisms of the storytelling and character development.'
- **R line (dr_residual)**, key `4973`
  - Gemma: "The review discusses the television show Parenthood and the reviewer's enjoyment of it."
  - Qwen: "The review discusses the reviewer's experience with the TV show Parenthood, highlighting their appreciation for its intelligent writing, realistic scenarios, and strong character development."
