"""Two-minute tour of juxtaposed on a synthetic tissue (runs in a few seconds)."""
import numpy as np
import juxtaposed as jx

rng = np.random.default_rng(0)
# 40 x 40 square lattice of spots, three domains: A | B | C (left to right)
xy = np.column_stack(np.unravel_index(np.arange(1600), (40, 40))).astype(float)
col = xy[:, 1]
domain_a, domain_b, domain_c = col < 13, (col >= 13) & (col < 27), col >= 27
noise = lambda: 0.3 * rng.normal(size=1600)
expr = np.column_stack([
    domain_a + noise(),        # geneA: marks domain A
    domain_b + noise(),        # geneB: marks domain B (touches A)
    domain_c + noise(),        # geneC: marks domain C (does not touch A)
    domain_a + noise(),        # geneA2: second marker of A (co-localized with geneA)
    noise(),                   # geneN: no structure
])
genes = ["geneA", "geneB", "geneC", "geneA2", "geneN"]

graph = jx.spatial_graph(xy, neighbors=4)
res = jx.juxtaposition(expr, graph, features=genes, scale=2)   # J + spectral-null p, q
print(res.table().round(3).to_string(index=False))
# geneA-geneB: J > 0 (exclusive domains that touch)
# geneA-geneC: J ~ 0 (exclusive, but no shared boundary)  <- Lee's L cannot tell these apart
# geneA-geneA2: J < 0 (same domain: concordant changes)

print(jx.compare(expr, graph, features=genes, scale=2).round(3).to_string(index=False))
