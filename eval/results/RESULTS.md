## Retrieval: pipeline stages

| method | main hit@1 | main MRR | hard hit@1 | hard hit@10 | hard MRR | latency (ms) |
|---|---|---|---|---|---|---|
| vector | 29/35 | 0.899 | 10/20 | 19/20 | 0.669 | 21 |
| BM25 | 31/35 | 0.914 | 11/20 | 19/20 | 0.695 | 3 |
| hybrid | 32/35 | 0.933 | 13/20 | 19/20 | 0.766 | 18 |
| vector + rerank | 33/35 | 0.961 | 16/20 | 18/20 | 0.838 | 498 |
| hybrid + rerank | 33/35 | 0.967 | 16/20 | 20/20 | 0.863 | 602 |

## Retrieval: ablations (hit@1 questions, MRR in brackets)

| config | vector main | vector hard | hybrid+rerank main | hybrid+rerank hard |
|---|---|---|---|---|
| base | 29/35 (0.899) | 10/20 (0.669) | 33/35 (0.967) | 16/20 (0.863) |
| c100 | 28/35 (0.872) | 11/20 (0.668) | 34/35 (0.977) | 15/20 (0.797) |
| c400 | 28/35 (0.866) | 9/20 (0.603) | 33/35 (0.967) | 15/20 (0.816) |
| nohead | 27/35 (0.843) | 8/20 (0.552) | 33/35 (0.967) | 16/20 (0.871) |
| minilm | 25/35 (0.811) | 11/20 (0.635) | 33/35 (0.967) | 16/20 (0.871) |

## Generation: prompt v1 vs v2

| | prompt v1 | prompt v2 |
|---|---|---|
| answerable questions run | 55 | 55 |
| wrongly refused | 1 ['h17'] | 0 [] |
| answers with a citation | 48/54 | 53/55 |
| answers citing the gold file | 46/54 | 52/55 |
| unanswerable refused | 19/19 | 19/19 |
