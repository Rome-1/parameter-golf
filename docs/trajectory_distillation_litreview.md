# Trajectory / Temporal Distillation for Parameter Golf

**Goal:** Overtrain a teacher model beyond the 10-minute budget (e.g. 30-60 min),
capture its latent-space features and training trajectory, then build auxiliary
loss terms that let a student model trained within the 10-minute budget recreate
those internal structures.

This is *temporal/trajectory distillation* rather than parameter distillation:
the teacher's value comes from its richer learned representations, not from
compressing its weights directly.

---

## 1. Foundational Knowledge Distillation

### Hinton, Vinyals & Dean (2015). "Distilling the Knowledge in a Neural Network." NeurIPS Workshop.
- **Key idea:** Train a student to match the teacher's softened output distribution
  (temperature-scaled softmax) via KL divergence. The "dark knowledge" in
  non-predicted class probabilities carries structural information about input
  similarities that hard labels lack.
- **Relevance:** Baseline output-level loss. Our scenario benefits from going
  *beyond* logit matching to intermediate representations, but soft-target KD is
  the foundation everything else builds on.
- arXiv: [1503.02531](https://arxiv.org/abs/1503.02531)

---

## 2. Trajectory Distillation (Teacher Checkpoints / Training Dynamics)

### Qin et al. (2022). "Efficient Knowledge Distillation from Model Checkpoints." NeurIPS.
- **Key idea:** An *intermediate* checkpoint (mid-training) is often a better
  teacher than the fully converged model, despite lower accuracy. A weak snapshot
  ensemble of checkpoints from a single run outperforms a strong ensemble of
  independently trained converged models.
- **Why:** Explained via the information bottleneck: intermediate representations
  have higher mutual information with the input and contain richer dark knowledge.
  Proposes an optimal checkpoint selection algorithm maximizing task-related MI.
- **Relevance:** Directly applicable. Our overtrained teacher produces a trajectory
  of checkpoints. Rather than distilling only from the final model, using
  mid-trajectory snapshots (or an ensemble thereof) may transfer more structure.
- arXiv: [2210.06458](https://arxiv.org/abs/2210.06458)

### Fujitsu Research (2026). "Following the Teacher's Footsteps: Scheduled Checkpoint Distillation for Domain-Specific LLMs."
- **Key idea:** Learning *how* the teacher learns matters more than learning *what*
  it learned. SCD systematically uses intermediate checkpoints during distillation,
  scheduling which checkpoint the student distills from at each stage of its own
  training, mimicking the teacher's trajectory.
- **Why:** Adaptive weighting preserves the student's inherent strengths on
  favorable subdomains while reducing its deficit on teacher-favored subdomains.
- **Relevance:** Concrete recipe for our scenario: overtrain the teacher, save
  checkpoints, then schedule the student to follow the teacher's learning path
  rather than just match its endpoint.
- arXiv: [2601.10114](https://arxiv.org/html/2601.10114v1)

### Cazenavette et al. (2022). "Dataset Distillation by Matching Training Trajectories." CVPR (Oral).
- **Key idea:** Synthesizes a small dataset such that models trained on it follow
  similar parameter-space trajectories as models trained on the full dataset.
  Optimizes synthetic data by backpropagating through student training steps.
- **Relevance:** Tangential but inspirational. Trajectory matching in parameter
  space could inspire auxiliary losses penalizing divergence between student and
  teacher training dynamics.
- arXiv: [2203.11932](https://arxiv.org/abs/2203.11932)

---

## 3. Progressive Distillation

### Salimans & Ho (2022). "Progressive Distillation for Fast Sampling of Diffusion Models." ICLR.
- **Key idea:** Distills a trained diffusion sampler (N steps) into a student
  (N/2 steps), then repeats iteratively. Each stage, the student learns to match
  2 teacher steps with 1 student step.
- **Relevance:** The progressive halving principle can be adapted: overtrain a
  teacher, distill to a medium model, distill to the budget-constrained student,
  with intermediate representations guiding each stage. The staged approach is
  analogous to our multi-checkpoint trajectory.
- arXiv: [2202.00512](https://arxiv.org/abs/2202.00512)

---

## 4. Feature-Matching Distillation

### Romero et al. (2015). "FitNets: Hints for Thin Deep Nets." ICLR.
- **Key idea:** Pioneer of feature-based distillation. Matches intermediate hidden
  representations (not just outputs) between teacher and student. A regressor
  layer bridges dimension mismatches.
- **Two-stage training:** (1) Pre-train student's early layers to match a teacher
  "hint" layer, (2) fine-tune entire student with KD loss on outputs.
- **Relevance:** Core template for our auxiliary losses. The hint-based
  pre-training maps to: pre-train student layers to match the overtrained
  teacher's intermediate representations, then fine-tune within budget.
- arXiv: [1412.6550](https://arxiv.org/abs/1412.6550)

### Zagoruyko & Komodakis (2017). "Paying More Attention to Attention." ICLR.
- **Key idea:** Defines spatial attention maps (aggregated absolute activations)
  as transferable knowledge. Student is trained to mimic the teacher's attention
  maps at multiple layers.
- **Relevance:** For transformers, this translates to matching self-attention
  distributions. An overtrained teacher's attention patterns encode structural
  knowledge about token relationships that a budget student can learn via
  auxiliary attention-matching losses.
- arXiv: [1612.03928](https://arxiv.org/abs/1612.03928)

### Yim et al. (2017). "A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning." CVPR.
- **Key idea:** Defines the "Flow of Solution Procedure" (FSP): knowledge
  captured as Gram matrices between feature maps of different layers,
  representing inter-layer information flow.
- **Relevance:** FSP captures relational structure between layers. For our
  transformer student, matching how information flows between layers in the
  overtrained teacher is a powerful auxiliary loss.
- CVPR PDF: [link](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf)

### Passalis & Tefas (2018). "Learning Deep Representations with Probabilistic Knowledge Transfer." ECCV.
- **Key idea:** PKT matches probability distributions of data in the feature
  space using kernel density estimation, rather than raw feature vectors.
  Minimizes divergence between distribution representations.
- **Relevance:** Distribution matching is more robust than point-wise feature
  matching when teacher and student have different capacities. Matching the
  *distribution* of the overtrained teacher's latent representations (rather
  than exact activations) may be more achievable for a smaller student.
- arXiv: [1803.10837](https://arxiv.org/abs/1803.10837)

### Chen et al. (2021). "Distilling Knowledge via Knowledge Review." CVPR.
- **Key idea:** ReviewKD: cross-stage connection paths where each teacher layer's
  features guide the student's corresponding layer *and all previous layers*.
- **Relevance:** The review mechanism could be particularly effective when the
  overtrained teacher has sophisticated hierarchical representations that a
  budget student needs to efficiently acquire.
- arXiv: [2104.09044](https://arxiv.org/abs/2104.09044)

---

## 5. Latent-Space Matching / Representation Alignment

### Kornblith et al. (2019). "Similarity of Neural Network Representations Revisited." ICML.
- **Key idea:** Introduced CKA (Centered Kernel Alignment) as a representation
  similarity metric. Operates on Gram matrices, dimension-agnostic, invariant to
  orthogonal transforms and isotropic scaling.
- **Relevance:** CKA provides the similarity metric for designing our auxiliary
  losses. Maximizing CKA between teacher and student hidden states is equivalent
  to minimizing a bound on MMD, giving non-vanishing gradients.
- arXiv: [1905.00414](https://arxiv.org/abs/1905.00414)

### Tian, Krishnan & Isola (2020). "Contrastive Representation Distillation." ICLR.
- **Key idea:** CRD formulates representation transfer as contrastive learning:
  positive pairs (same input, teacher vs student embeddings) are pulled together,
  negative pairs pushed apart. Captures structural information KL divergence misses.
- **Relevance:** CRD transfers the *structure* of the teacher's representation
  space. For our overtrained teacher, CRD-style losses encourage the student to
  develop an internally coherent latent space that mirrors the teacher's, even if
  absolute representations differ.
- arXiv: [1910.10699](https://arxiv.org/abs/1910.10699)

### Park et al. (2023). "Feature Structure Distillation with CKA in BERT Transferring." Expert Systems with Applications.
- **Key idea:** Decomposes feature structure into intra-feature, local
  inter-feature, and global inter-feature structures. Implements CKA-based
  losses for each. Uses memory-augmented transfer with clustering for global
  structures.
- **Relevance:** Closest existing work to our specific setup for language models.
  The tri-level decomposition provides a concrete taxonomy for designing
  auxiliary losses.
- [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417423014823)

### Zhang et al. (2024). "Dual-Space Knowledge Distillation for LLMs." EMNLP.
- **Key idea:** DSKD identifies that teacher and student output spaces are
  misaligned. Projects hidden states into each other's representation spaces
  with learned projectors, plus cross-model attention for vocabulary alignment.
- **Relevance:** The projector approach is critical when teacher and student have
  different hidden dimensions.
- arXiv: [2406.17328](https://arxiv.org/abs/2406.17328)

---

## 6. Temporal / Self-Distillation Over Training Time

### Furlanello et al. (2018). "Born-Again Neural Networks." ICML.
- **Key idea:** Students with identical architecture to their teachers, iterated
  over generations, consistently outperform the teacher. Dark knowledge from
  non-predicted classes provides compounding regularization.
- **Relevance:** Even without compression, distillation from an overtrained model
  provides superior training signal. A single-generation BAN with the overtrained
  teacher is our simplest baseline.
- arXiv: [1805.04770](https://arxiv.org/abs/1805.04770)

### Zhang et al. (2019). "Be Your Own Teacher: Self-Distillation." ICCV.
- **Key idea:** Deeper layers act as teachers for shallower layers via KL
  divergence on outputs plus L2 loss on feature maps. Single-run self-distillation.
- **Relevance:** During the student's budget-constrained training, later layers
  (which converge faster) can provide self-distillation signal to earlier layers,
  complementing the external teacher signal.
- ICCV PDF: [link](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Be_Your_Own_Teacher_Improve_the_Performance_of_Convolutional_Neural_ICCV_2019_paper.pdf)

---

## 7. Distillation for LM Pre-training and Compression

### Sun et al. (2019). "Patient Knowledge Distillation for BERT." EMNLP.
- **Key idea:** Student "patiently" learns from multiple intermediate teacher
  layers (PKD-Skip: every k-th layer; PKD-Last: last k layers). Uses MSE on
  normalized [CLS] hidden states.
- **Relevance:** Concrete recipe for intermediate-layer matching in transformers.
  PKD-Skip is especially applicable when teacher and student have different depths.
- arXiv: [1908.09355](https://arxiv.org/abs/1908.09355)

### Jiao et al. (2020). "TinyBERT: Distilling BERT for NLU." EMNLP Findings.
- **Key idea:** Two-stage distillation matching embeddings, attention matrices,
  and hidden states. TinyBERT4 achieves 96.8% of BERT-Base at 7.5x smaller.
- **Relevance:** The multi-component loss (embedding + attention + hidden state)
  is the most complete template for our auxiliary loss design.
- arXiv: [1909.10351](https://arxiv.org/abs/1909.10351)

### Gu et al. (2024). "MiniLLM: Knowledge Distillation of Large Language Models." ICLR.
- **Key idea:** Replaces forward KLD with reverse KLD for generative LM
  distillation. Forward KLD forces the student to cover all teacher modes
  (including low-probability ones). Reverse KLD produces mode-seeking behavior
  better suited to autoregressive generation.
- **Relevance:** For our scenario, the choice of divergence direction matters.
  Reverse KLD avoids forcing the smaller student to cover unlikely modes of the
  overtrained teacher.
- arXiv: [2306.08543](https://arxiv.org/abs/2306.08543)

### Ko et al. (2024). "DistiLLM: Streamlined Distillation for LLMs." ICML.
- **Key idea:** Skew KL divergence (interpolation between forward and reverse
  KLD) with theoretical convergence guarantees. Combined with adaptive off-policy
  training.
- **Relevance:** Tunable middle ground between forward and reverse KLD for our
  budget-constrained setting.
- arXiv: [2402.03898](https://arxiv.org/abs/2402.03898)

### Liang et al. (2023). "Less is More: Task-aware Layer-wise Distillation." ICML.
- **Key idea:** TED: task-aware filters select which knowledge from teacher hidden
  representations is useful, filtering redundant information before alignment.
  Addresses under-fitting when students cannot absorb everything from the teacher.
- **Relevance:** Directly addresses our capacity constraint. Filtering teacher
  signal prevents the student from wasting capacity on irrelevant features.
- arXiv: [2210.01351](https://arxiv.org/abs/2210.01351)

### Yao et al. (2025). "MiniPLM: Knowledge Distillation for Pre-Training LMs." ICLR.
- **Key idea:** Distills teacher knowledge into the training *data distribution*
  via offline "difference sampling." Reduces pre-training data demand by 2.4x.
- **Relevance:** Alternative to representation-matching: reshape training data to
  implicitly encode teacher knowledge. Could be combined with latent-space losses.
- arXiv: [2410.17215](https://arxiv.org/abs/2410.17215)

---

## 8. Compute-Optimal Distillation / Overtraining-then-Distilling

### Busbridge et al. (2025). "Distillation Scaling Laws." ICML.
- **Key idea:** Derives scaling laws predicting distilled model performance as a
  function of compute budget and allocation between teacher and student training.
- **Key findings:**
  1. Distillation beats supervised learning only when total compute/tokens is
     below a student-size-dependent threshold, or when a teacher already exists.
  2. Overtraining the teacher becomes more efficient as the number of student
     tokens increases.
  3. The "capacity gap" reflects a learning capacity gap, not just size mismatch.
- **Relevance:** Critically relevant. Directly models our scenario. Tells us
  *how much* to overtrain the teacher for a given student budget, and when
  distillation is worth the teacher training cost.
- arXiv: [2502.08606](https://arxiv.org/abs/2502.08606)

---

## Synthesis: Recommended Approach for Parameter Golf

Based on this review, the most promising strategy for our constraint (16MB
artifact, 10-minute training budget on 8xH100):

### Phase 1: Overtrain Teacher (Offline, Unlimited Compute)
- Train the baseline architecture for 30-60 minutes (3-6x budget)
- Save checkpoints every N steps (trajectory)
- Extract and cache per-layer hidden states on a representative data subset

### Phase 2: Student Training (Within 10-min Budget)
Combine these auxiliary losses with the standard cross-entropy:

1. **Soft-target KD** (Hinton): KL divergence on temperature-softened logits.
   Use reverse or skew KLD (MiniLLM/DistiLLM) to avoid mode-covering.

2. **Hidden-state matching** (FitNets/TinyBERT): MSE or cosine similarity
   between student and teacher intermediate representations at mapped layers.
   Use learned linear projectors if dimensions differ.

3. **CKA structural alignment** (Kornblith/Park): Match representation *structure*
   rather than point-wise activations. More robust when student has less capacity.

4. **Checkpoint scheduling** (Qin/SCD): Schedule which teacher checkpoint the
   student distills from at each stage of training. Early student matches early
   teacher; late student matches final teacher.

5. **Task-aware filtering** (TED): Filter which teacher representations to match,
   preventing under-capacity student from wasting budget on irrelevant features.

### Loss Function
```
L_total = L_CE(student, labels)
        + alpha * L_KD(student_logits, teacher_logits, T)
        + beta  * L_hidden(student_hiddens, teacher_hiddens)
        + gamma * L_CKA(student_features, teacher_features)
```

Where alpha, beta, gamma are search hyperparameters, and teacher states may come
from different checkpoints at different training stages.

### Key Considerations for pgolf
- **Budget:** Teacher features must be pre-computed and cached. No live teacher
  forward pass during the 10-minute student training.
- **Storage:** Cached teacher hidden states could be large. May need to subsample
  or compress.
- **Artifact size:** Auxiliary loss code adds to the code bytes (counts toward
  16MB). Keep implementation minimal.
- **Quantization:** Distillation could help mitigate quantization error, as
  shown by the DepthRecurrence submission's Noisy QAT approach.

---

## References

1. Hinton et al. (2015). arXiv:1503.02531
2. Qin et al. (2022). arXiv:2210.06458
3. Fujitsu (2026). arXiv:2601.10114
4. Cazenavette et al. (2022). arXiv:2203.11932
5. Salimans & Ho (2022). arXiv:2202.00512
6. Romero et al. (2015). arXiv:1412.6550
7. Zagoruyko & Komodakis (2017). arXiv:1612.03928
8. Yim et al. (2017). CVPR 2017
9. Passalis & Tefas (2018). arXiv:1803.10837
10. Chen et al. (2021). arXiv:2104.09044
11. Kornblith et al. (2019). arXiv:1905.00414
12. Tian et al. (2020). arXiv:1910.10699
13. Park et al. (2023). Expert Systems with Applications
14. Zhang et al. (2024). arXiv:2406.17328
15. Furlanello et al. (2018). arXiv:1805.04770
16. Zhang et al. (2019). ICCV 2019
17. Sun et al. (2019). arXiv:1908.09355
18. Jiao et al. (2020). arXiv:1909.10351
19. Gu et al. (2024). arXiv:2306.08543
20. Ko et al. (2024). arXiv:2402.03898
21. Liang et al. (2023). arXiv:2210.01351
22. Yao et al. (2025). arXiv:2410.17215
23. Busbridge et al. (2025). arXiv:2502.08606
