# Prior-art notes for the LLM-synthesis question (quick search, 3 queries; not an audit)
Found, closest first (secondary sources unless marked):
* **Weight-space programming** (blog-level, Nov 2026; DataCamp summary https://www.datacamp.com/blog/weight-space-programming): a general-purpose LLM reads/edits/writes network weights from a plain-language goal without gradient descent, mostly small networks, one LLM, not peer-reviewed. Critique: https://www.seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights . **Read the primary post before positioning.** Closest to category 4.
* **NeuroGen** (arXiv 2505.12470): trained LLM generating network parameters directly; ~70-90% of baseline performance. Different category (trained generator).
* **Drag-and-Drop LLMs** (arXiv 2506.16406): prompt-to-LoRA-weights hypernetwork. Trained generator.
* **TracrBench** (arXiv 2409.13714): LLMs write RASP programs compiled by Tracr; GPT-4-turbo correct on 57/101 programs (20-shot, best-of-5). Category 3a.
* **torchwright** (https://dev.to/breachprotocol/torchwright-builds-working-transformer-weights-from-python-no-training-involved-5c0h): compiler from a typed computation graph to transformer weights. Category 1.
* Overlaps with earlier claims: hand-built SHA transformer write-up reports Tracr numerical instability beyond ~400 tokens (https://andrew.gr/stories/sha/) vs our B1/B2; analog-hardware transformers, e.g. ABNAT (arXiv 2610.22065): weight perturbations disrupt attention and accumulate with depth; noise-aware training (vs our G1, C2).
Not found: a comparison of parameter-perturbation robustness across compiler-built, analytic, LLM-synthesized and trained networks (absence is weak evidence: 3 searches).
