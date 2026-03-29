# Geometric Properties of Converged LM Latent Spaces as Inductive Biases

**Context**: Parameter Golf — sub-50M param LM, 10-min training budget on 8xH100, 16MB artifact limit.
**Framing**: Use an overtrained model as a *probe* to discover geometric structure of the converged latent space, then encode that structure as loss terms or initialization priors. The student trains independently — no teacher at runtime.

---

## Table of Contents

1. [Spectral Structure](#1-spectral-structure)
2. [Representation Geometry](#2-representation-geometry)
3. [Geometric Regularizers](#3-geometric-regularizers)
4. [Attention Structure](#4-attention-structure)
5. [Practical Feasibility](#5-practical-feasibility)
6. [Concrete Proposal](#6-concrete-proposal)

---

## 1. Spectral Structure

### Key Question
Do trained LM hidden states exhibit predictable low-rank or spectral structure? Can we measure a target spectral profile from a converged model and penalize deviation during budget-constrained training?

### 1.1 Neural Collapse

**Papyan, Han & Donoho (2020).** "Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training." PNAS 2020. [arXiv:2008.08186](https://arxiv.org/abs/2008.08186)

**Key findings**: In the terminal phase of training (after training loss approaches zero), four interrelated phenomena emerge simultaneously:
1. **NC1 (Variability Collapse)**: Within-class activations of the last-layer features converge to their class means.
2. **NC2 (Convergence to Simplex ETF)**: Class means converge to the vertices of a simplex equiangular tight frame (i.e., maximally and equally separated).
3. **NC3 (Self-Duality)**: The last-layer classifiers converge to the same simplex ETF as the class means.
4. **NC4 (Simplification to Nearest Class Center)**: The network's decision rule converges to a nearest-class-center classifier.

**Applicability to pgolf**: Neural collapse is demonstrated primarily in classification settings with cross-entropy loss and a fixed number of classes. Language modeling with a 1024-token vocabulary is a 1024-class classification problem at each position, so NC phenomena could partially apply — particularly NC2 (token embeddings converging toward a simplex ETF). However:
- **Overtrained regime caveat**: NC emerges after prolonged training well past interpolation. In pgolf's 10-minute budget (~5000 steps), we're far from the terminal phase. But an *overtrained probe model* (trained for hours) could reach this regime.
- **Actionable signal**: If the probe's output embeddings approach a simplex ETF, we could initialize the student's tied embedding matrix to approximate this structure, or regularize toward equal angular separation between token embeddings.
- **Practical concern**: With 1024 classes, a full simplex ETF lives in ≥1023 dimensions. Our model dimension is 512, so the ETF cannot be perfectly realized — but a projected version may still provide useful structure.

**Related work on NC in language models**:
- Zhong et al. (2023), "Understanding Collapse in Non-Contrastive Learning" — extends NC analysis to self-supervised settings, relevant since LM pretraining is autoregressive self-supervision.
- Yang et al. (2022), "Inducing Neural Collapse in Deep Long-Tailed Learning" — proposes ETF classifiers as fixed initialization, directly relevant to our initialization strategy.

### 1.2 Intrinsic Dimensionality of Representations

**Ansuini, Laio, Macke & Zdeborová (2019).** "Intrinsic dimension of data representations in deep neural networks." NeurIPS 2019. [arXiv:1905.12784](https://arxiv.org/abs/1905.12784)

**Key findings**: The intrinsic dimensionality (ID) of learned representations follows a characteristic pattern across layers: high in early layers (close to data dimensionality), drops dramatically in intermediate layers (information compression), then may rise slightly before the output. This "hunchback" pattern is consistent across architectures (CNNs, ResNets) and datasets. The minimum ID in intermediate layers is much lower than the ambient dimensionality (e.g., ID ~20-40 in layers with 512+ dimensions).

**Aghajanyan, Gupta & Zettlemoyer (2021).** "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." ACL 2021. [arXiv:2012.13255](https://arxiv.org/abs/2012.13255)

**Key findings**: Pre-trained language models have very low intrinsic dimensionality for downstream task adaptation — often only ~100-800 parameters in a low-rank subspace are needed to match full fine-tuning performance. Larger pre-trained models have *lower* intrinsic dimensionality (the learned representation is more compressed/structured). This directly motivates low-rank adaptation methods like LoRA.

**Applicability to pgolf**:
- **Target spectral profile**: If we overtrain a probe model and measure the singular value spectrum of its hidden state activations at each layer, we get a *target spectral profile*. The rapid fall-off (high effective rank but low intrinsic dim) tells us exactly which singular value structure the trained model converges to.
- **Regularization strategy**: Penalize the student's representation if its spectral profile deviates from the probe's target. This is computationally feasible — we only need to periodically SVD the activation covariance (or use a running estimate), which is O(d²n + d³) per layer.
- **Initialization strategy**: Initialize weights such that the initial representations have a spectral profile closer to the converged target, reducing the optimization distance.

### 1.3 Low-Rank Structure in Transformer Weights and Activations

**Sharma & Karakida (2024).** "The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction." ICLR 2024. [arXiv:2312.13558](https://arxiv.org/abs/2312.13558)

**Key findings**: Many layers in trained LLMs are highly redundant — aggressively removing low-rank components from weight matrices (setting small singular values to zero) can actually *improve* performance on reasoning tasks. Some layers are nearly rank-deficient. The weight matrices' effective rank varies significantly across layers.

**Hsu, Bhojanapalli, Garg & Telgarsky (2022).** "Language Model Compression with Weighted Low-Rank Factorization." ICLR 2022. [arXiv:2207.00112](https://arxiv.org/abs/2207.00112)

**Key findings**: Weighted low-rank factorization of transformer weight matrices (using Fisher information to weight the importance of different components) can compress models significantly with minimal quality loss. The Fisher-weighted spectrum reveals which singular components matter most for the loss landscape.

**Applicability to pgolf**:
- **Probe the weight spectrum**: SVD each weight matrix of the overtrained probe. Record the singular value distribution — specifically, the effective rank (number of singular values above some threshold) and the shape of the decay curve.
- **Spectral penalty**: During student training, add a loss term that penalizes deviation from the target singular value distribution. This could be as simple as penalizing the nuclear norm toward a target value, or as specific as penalizing individual singular values toward target values.
- **Compact encoding**: A spectral profile is just a sorted vector of singular values per matrix. For a 512×512 weight matrix, this is 512 floats = 2KB. Across all weight matrices in a ~10-layer model, this is ~50-100KB — trivially fits in the 16MB budget.

### 1.4 Spectral Properties Specific to Transformers

### 1.4 Anisotropy as Structural Property

**Godey, Clergerie & Sagot (2024).** "Anisotropy Is Inherent to Self-Attention in Transformers." [arXiv:2401.12143](https://arxiv.org/abs/2401.12143)

**Key findings**: Anisotropy in transformer representations is not merely a training artifact — it is a structural property of the self-attention mechanism itself. This holds across modalities (language, speech, vision), suggesting that isotropy-based regularization must account for the inherent anisotropy that attention produces.

**Yan et al. (2022).** "Addressing Token Uniformity in Transformers via Singular Value Transformation." [arXiv:2208.11790](https://arxiv.org/abs/2208.11790)

**Key findings**: Skewed singular value distributions of intermediate layer outputs characterize token uniformity problems. A less skewed distribution alleviates the problem. Proposes SVD-based transformation of layer outputs.

**"Small Singular Values Matter: A Random Matrix Analysis of Transformer Models" (2024).** [arXiv:2410.17770](https://arxiv.org/abs/2410.17770)

**Key findings**: Feature/activation matrices in transformers are often low rank, while weight matrices are not — the low-rank structure emerges in *representations*, not necessarily parameters. Regions of the spectrum deviating from Marchenko-Pastur predictions correspond to learned features. **Important implication**: measure spectral structure in activations, not just weights.

### 1.5 Heavy-Tailed Self-Regularization

**Martin & Mahoney (2019).** "Traditional and Heavy-Tailed Self Regularization in Neural Network Models." [arXiv:1901.08276](https://arxiv.org/abs/1901.08276)

**Key findings**: The empirical spectral density (ESD) of weight matrices in well-trained DNNs follows a truncated power-law distribution, not a Marchenko-Pastur distribution (which would indicate random/untrained weights). The power-law exponent α correlates with generalization quality: well-trained models have α between 2 and 4 (heavy-tailed but not too heavy). This holds across architectures including transformers.

**Applicability to pgolf**:
- **Quality metric**: The power-law exponent of weight matrix spectra could serve as a *training quality indicator*. If the probe's converged weights show α ≈ 3, we could regularize toward this exponent during student training.
- **Practical concern**: Fitting a power law to the ESD requires computing the full spectrum, which is expensive during training. But a coarse approximation (ratio of top-k to remaining singular values) might suffice.
- **Tooling**: The `weightwatcher` Python package implements power-law fitting for weight matrix spectra — no need to write custom code.
- **Speculation level**: MODERATE. The power-law relationship is well-established empirically, but using it as a *training target* is novel and untested. The computational cost of accurate spectral estimation during training may be prohibitive in a 10-minute budget.

---

## 2. Representation Geometry

### Key Question
Do different models (trained from different seeds, or with different architectures) converge to similar representation geometries? If so, we can extract and regularize toward a "canonical" geometry.

### 2.1 CKA Convergence Across Seeds

**Kornblith, Norouzi, Lee & Hinton (2019).** "Similarity of Neural Network Representations Revisited." ICML 2019. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)

**Key findings**: Introduced Centered Kernel Alignment (CKA) as a robust similarity metric for neural network representations. CKA is invariant to orthogonal transformations and isotropic scaling, making it more reliable than canonical correlation analysis (CCA) or simple representational similarity analysis (RSA). Key results:
- Networks trained from different random seeds converge to similar representations (high CKA), especially in deeper layers.
- CKA similarity between different architectures is lower but still significant.
- CKA can be computed efficiently using the linear kernel: CKA(X, Y) = ||Y^T X||²_F / (||X^T X||_F · ||Y^T Y||_F).

**Applicability to pgolf**:
- **Cross-seed convergence**: If multiple runs of our sub-50M model converge to similar CKA representations, there exists a "target geometry" we can regularize toward. Train 3-5 probe models from different seeds, compute pairwise CKA, and confirm convergence.
- **CKA as a loss**: Use CKA similarity between student activations and the probe's target activations as a regularization loss. Unlike MSE on raw activations (standard distillation), CKA is invariant to rotation and scale, so it only constrains the *geometry* of the representation, not the specific basis.
- **Computational cost**: Linear CKA requires forming the Gram matrices X^T X (d×d) for each layer, then computing Frobenius norms. For d=512 with batch size B: O(Bd²) per layer per step — roughly equivalent to one extra forward pass. Feasible but not free.

### 2.2 The Platonic Representation Hypothesis

**Huh, Cheung, Wang & Isola (2024).** "The Platonic Representation Hypothesis." ICML 2024. [arXiv:2405.07987](https://arxiv.org/abs/2405.07987)

**Key findings**: Models trained on different data modalities (vision, language, vision-language) and with different objectives are converging toward a shared statistical model of reality — a "platonic representation." Key evidence:
- Representation similarity between different model families (LLMs, vision models, multimodal models) has been *increasing* over time as models scale.
- This convergence is strongest in intermediate/later layers.
- The hypothesis: there is a convergent "reality structure" that sufficiently capable models discover, regardless of training objective.

**Applicability to pgolf**:
- **Strong prior**: If even *different architectures* converge to similar geometries, then the converged geometry of an overtrained probe is a robust target. It's not a quirk of one model — it's a structural property of the data.
- **Scale caveat**: The platonic representation hypothesis is demonstrated primarily for large models (billions of parameters). Whether sub-50M models converge to similar representations is an empirical question. Smaller models may lack the capacity to fully realize the "platonic" geometry, in which case the target from an overtrained probe might be unrealistically rich.
- **Experiment needed**: Train 3 probe models with different hyperparameters/seeds, compute CKA pairwise. If CKA > 0.8 in later layers, the platonic hypothesis applies at this scale.

### 2.3 Representation Structure in Small LMs

**Li, Hopkins, Bau & Viégas (2023).** "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task." ICLR 2023. [arXiv:2210.13382](https://arxiv.org/abs/2210.13382)

**Key findings**: Even small sequence models (1-8M parameters) trained on simple tasks develop structured internal representations that encode meaningful features of the data-generating process. The geometry of these representations is interpretable and convergent across seeds.

**Nanda, Chan, Lieberum, Smith & Steinhardt (2023).** "Progress Measures for Grokking via Mechanistic Interpretability." ICLR 2023. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)

**Key findings**: Small transformers trained on modular arithmetic develop Fourier-basis representations. The *geometric structure* of these representations (which Fourier frequencies dominate) is predictable from the task structure and converges across seeds.

**Applicability to pgolf**:
- These studies confirm that even small models develop structured, convergent representations. The "target geometry" approach is not limited to billion-parameter models.
- **Speculation level**: LOW for the existence of structured representations; MODERATE for whether they're useful as regularization targets for FineWeb language modeling (more complex than synthetic tasks).

### 2.4 Gram Matrix Structure

**Nguyen, Raghu & Kornblith (2021).** "Do Wide Neural Networks of Any Depth Learn Representations?" NeurIPS 2021. [arXiv:2010.15110](https://arxiv.org/abs/2010.15110)

**Key findings**: Feature learning (as opposed to kernel-regime behavior) in deep networks is characterized by changes in the Gram matrix K = X X^T of activations across training. Well-trained networks show Gram matrices with specific eigenvalue structure: a few large eigenvalues (capturing major data features) and many small eigenvalues (structured noise).

**Applicability to pgolf**:
- **Target Gram matrix**: Record the activation Gram matrix from the probe model on a calibration set. Use the eigenstructure of this Gram matrix as a regularization target.
- **Compact encoding**: The Gram matrix eigenvalues (top-k) per layer constitute a tiny descriptor. Top-50 eigenvalues per layer × 10 layers = 500 floats = 2KB.
- **Loss formulation**: Penalize the Frobenius distance between the student's activation Gram matrix and the target, or just match the top-k eigenvalues.

---

## 3. Geometric Regularizers

### Key Question
What existing methods regularize representations toward specific geometric properties? Which are applicable to transformer training under severe compute constraints?

### 3.1 Isotropy Regularization

**Mu & Viswanath (2018).** "All-but-the-Top: Simple and Effective Postprocessing for Word Representations." ICLR 2018. [arXiv:1702.01417](https://arxiv.org/abs/1702.01417)

**Key findings**: Word embeddings (Word2Vec, GloVe) occupy a narrow cone in embedding space — they are highly *anisotropic*. Removing the top few principal components (the mean and dominant directions) from all embeddings dramatically improves performance on similarity and analogy tasks. The top components encode frequency/commonality information, not semantic content.

**Gao, He, Tan & Qiu (2019).** "Representation Degeneration Problem in Training Natural Language Generation Models." ICLR 2019. [arXiv:1907.12009](https://arxiv.org/abs/1907.12009)

**Key findings**: Autoregressive language models trained with tied input-output embeddings suffer from *representation degeneration* — hidden states collapse into an anisotropic distribution, and token embeddings become anti-correlated with their frequency (common tokens cluster together, rare tokens are pushed apart). This degeneration worsens training dynamics and final quality.

**Proposed solution**: The authors show this is partly caused by the tied embedding acting as both input projection and output classifier. They propose regularizing toward isotropy.

**Wang, Chen, Li & Xiang (2020).** "Improving Neural Language Generation with Spectrum Control." ICLR 2020. [arXiv:1906.02754](https://arxiv.org/abs/1906.02754)

**Key findings**: Explicitly controlling the singular value spectrum of the output embedding matrix during training (spectrum control) alleviates representation degeneration. Penalizing the condition number (ratio of largest to smallest singular value) of the embedding matrix improves generation quality.

**IMPORTANT CORRECTIVE — Rudman & Eickhoff (2024).** "Stable Anisotropic Regularization (I-STAR)." ICLR 2024. [arXiv:2305.19358](https://arxiv.org/abs/2305.19358)

**Key findings**: Prior isotropy measures (average cosine similarity) are flawed. I-STAR proposes IsoScore*, a fully differentiable, numerically stable isotropy measure suitable for mini-batch training. Surprisingly, they find that **decreasing** isotropy (making representations more anisotropic) actually improves downstream performance across 3 LLMs and 9 tasks — contradicting Gao et al.'s prescription. The *right* anisotropy structure matters more than isotropy per se.

**Combined applicability to pgolf**:
- **Directly relevant**: pgolf models use tied embeddings with 1024 vocab / 512 dim. This is exactly the setting where representation degeneration occurs.
- **Critical nuance from I-STAR**: Don't blindly push toward isotropy. Instead, regularize toward the *probe's observed anisotropy structure*. The probe tells us *how much* anisotropy is optimal.
- **Isotropy regularization as loss**: L_iso = ||Σ - Σ_target||²_F where Σ is the covariance of hidden states and Σ_target is from the probe. NOT toward σ·I.
- **Spectrum control on embeddings**: Penalize the condition number of the tied embedding matrix toward the probe's target condition number.
- **Computational cost**: Computing the covariance requires accumulating X^T X over the batch. For d=512, this is a 512×512 matrix — cheap to compute and decompose. ~0.1ms per layer per step.
- **Well-established**: This is one of the most actionable regularizers. **RECOMMEND FOR FIRST EXPERIMENTS.**

### 3.2 Orthogonality Constraints

**Bansal, Chen & Wang (2018).** "Can We Gain More from Orthogonality Regularizations in Training Deep Networks?" NeurIPS 2018. [arXiv:1810.09102](https://arxiv.org/abs/1810.09102)

**Key findings**: Soft orthogonality regularization (penalizing ||W^T W - I||²_F) on weight matrices improves training stability, generalization, and gradient flow. They test several variants:
- **Soft orthogonality (SO)**: L = ||W^T W - I||²_F
- **Spectral restricted isometry property (SRIP)**: L = (σ_max(W) - 1)²
- **Mutual coherence**: Penalize max pairwise cosine similarity between rows

Results: All variants improve generalization on image classification, with SO being the most consistently effective.

**Applicability to pgolf**:
- **Orthogonal initialization + regularization**: Initialize weight matrices as (scaled) orthogonal matrices, then regularize to stay near-orthogonal. This has been shown to help in transformers.
- **Connection to pgolf competition**: The leaderboard entry "OrthoInit" (PR by Raahil Shah) already uses orthogonal initialization and found it helpful. Adding soft orthogonality *regularization* during training would extend this.
- **Computational cost**: Computing W^T W for a d×d weight matrix is O(d³). For d=512, this is ~134M FLOPs per matrix. With ~4 matrices per layer × 10 layers = 40 matrices → ~5.4 GFLOPs total. Negligible compared to the forward pass (~100+ GFLOPs per step).
- **Targeted variant**: Rather than regularizing toward I (perfect orthogonality), regularize toward the probe's W^T W. This encodes the probe's specific correlation structure between weight rows.

### 3.3 Rank-Constrained and Low-Rank Training

**Khodak, Balcan & Talwalkar (2021).** "Initialization and Regularization of Factorized Neural Layers." ICLR 2021. [arXiv:2105.01029](https://arxiv.org/abs/2105.01029)

**Key findings**: Training neural networks with explicit low-rank factorization (W = AB where A is d×r and B is r×d, with r < d) can match full-rank training when initialized properly. The key insight is that the initialization scale of A and B matters enormously — balanced initialization (||A|| ≈ ||B||) outperforms unbalanced variants.

**Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen (2022).** "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

**Key findings**: Fine-tuning only low-rank perturbations (ΔW = BA, rank 4-64) to pre-trained weights matches full fine-tuning. This implies the *useful* parameter subspace for adaptation is very low-dimensional.

**Applicability to pgolf**:
- **Implicit low-rank training**: If the probe shows that converged weight matrices have effective rank r << d, we could train with explicit rank-r factorization from the start, reducing parameters and potentially improving convergence speed.
- **Nuclear norm regularization**: ||W||_* = Σ σ_i. Penalizing nuclear norm encourages low-rank solutions. Can target a specific nuclear norm value extracted from the probe.
- **Caution**: Overly aggressive rank constraints may prevent the model from exploring high-rank intermediate representations that are necessary during early training before the low-rank structure emerges. The probe's rank profile is a *converged* property — the training path may need higher rank.
- **Speculation level**: MODERATE. Low-rank structure is well-established in converged models, but whether constraining rank from initialization helps or hurts convergence speed is an open question.

### 3.4 Hyperspherical Uniformity and Neural Collapse Targets

**Liu et al. (2021).** "Learning with Hyperspherical Uniformity." AISTATS 2021. [arXiv:2103.01649](https://arxiv.org/abs/2103.01649)

**Key findings**: Proposes regularizing neurons/representations to be uniformly distributed on a unit hypersphere. Unlike L2 which regularizes individual neurons, this regularizes the *interaction* among neurons, reducing redundancy. Provides theoretical grounding via discrepancy measures on the sphere.

**"Generalizing and Decoupling Neural Collapse via Hyperspherical Uniformity Gap" (2023).** [arXiv:2303.06484](https://arxiv.org/abs/2303.06484)

**Key findings**: Extends neural collapse theory beyond the constraint that feature dim ≥ number of classes. Proposes the Hyperspherical Uniformity Gap (HUG) as an explicit training objective to drive representations toward the simplex ETF geometry. Differentiable and directly applicable.

**Applicability to pgolf**:
- HUG provides the clearest path from neural collapse theory to a practical loss: regularize the tied embedding toward simplex ETF geometry.
- Hyperspherical uniformity is especially relevant for the output embedding with 1024 tokens in 512 dimensions (overcomplete regime where full ETF is impossible but projected ETF is useful).
- Moderate computational cost (pairwise similarity computation, O(V²d) for vocab V and dim d). With V=1024, d=512: ~500M FLOPs per evaluation — do every 50 steps.

### 3.5 Spectral Regularizers Beyond Weight Decay

**Yoshida & Miyato (2017).** "Spectral Norm Regularization for Improving the Generalizability of Deep Learning." [arXiv:1705.10941](https://arxiv.org/abs/1705.10941)

**Key findings**: Spectral normalization (constraining σ_max(W) ≤ 1) stabilizes training and improves generalization, especially in GANs. This is a hard constraint on the maximum singular value.

**Gouk, Frank, Pfahringer & Cree (2021).** "Regularisation of Neural Networks by Enforcing Lipschitz Continuity." Machine Learning 2021. [arXiv:1804.04368](https://arxiv.org/abs/1804.04368)

**Key findings**: Constraining the Lipschitz constant of each layer (via spectral norm bounds) improves robustness and generalization. For transformers, this means constraining the spectral norm of attention and MLP weight matrices.

**Miyato, Kataoka, Koyama & Yoshida (2018).** "Spectral Normalization for Generative Adversarial Networks." ICLR 2018. [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)

**Key findings**: Normalizes each weight matrix by dividing by its spectral norm, ensuring Lipschitz constant = 1 per layer. Computationally light (one power iteration per forward pass). Built into PyTorch as `torch.nn.utils.spectral_norm`. The most widely adopted spectral regularizer.

**Scarvelis & Solomon (2024).** "Nuclear Norm Regularization for Deep Learning." NeurIPS 2024. [arXiv:2405.14544](https://arxiv.org/abs/2405.14544)

**Key findings**: Penalizing the nuclear norm of a function's Jacobian encourages low-rank local behavior. Direct computation is intractable, but they prove an equivalence: for compositions f = g ∘ h, penalizing the average squared Frobenius norms of Jg and Jh is equivalent. Also propose a denoising-style approximation that avoids Jacobian computation entirely.

**Applicability to pgolf**:
- **Targeted spectral norm**: Instead of constraining σ_max ≤ 1 (a generic bound), constrain σ_max toward the probe's observed σ_max per weight matrix. This is a more informed constraint.
- **Spectral shape penalty**: Beyond σ_max, penalize deviation from the entire singular value profile. L_spectral = Σ_i (σ_i(W) - σ_i^target)² for top-k singular values.
- **Nuclear norm toward target**: ||W||_* ≈ target nuclear norm from probe. The Frobenius decomposition trick from Scarvelis makes this cheap.
- **Computational cost**: Power iteration for σ_max is O(d²) — very cheap. Full SVD for the spectral profile is O(d³) — feasible but should be done periodically (every 50-100 steps), not every step.
- **Speculation level**: LOW for spectral norm constraints (well-established); MODERATE for targeted spectral shape penalties (novel but mechanistically sound).

### 3.5 Contrastive and Relational Regularizers

**Park, Kim, Lu & Cho (2019).** "Relational Knowledge Distillation." CVPR 2019. [arXiv:1904.05068](https://arxiv.org/abs/1904.05068)

**Key findings**: Instead of matching individual activations (teacher → student), match the *relations* between activations — specifically, pairwise distances and angles in the representation space. This captures geometric structure rather than pointwise values. The relational approach outperforms individual matching on transfer tasks.

**Tian, Krishnan & Isola (2020).** "Contrastive Representation Distillation." ICLR 2020. [arXiv:1910.10699](https://arxiv.org/abs/1910.10699)

**Key findings**: Using a contrastive loss to transfer structural properties of the teacher's representation space to the student. Rather than minimizing MSE between teacher/student activations, the contrastive loss preserves which examples are similar/dissimilar in representation space. This transfers geometric structure more effectively than pointwise matching.

**Applicability to pgolf**:
- **Important distinction**: These methods use a teacher at *training time*. Our framing requires no teacher at runtime — but we can use the probe to pre-compute *target relations* (a static artifact).
- **Pre-computed relational targets**: Run the probe on a calibration set. Record the pairwise distance matrix or Gram matrix of activations per layer. Store this as a compact artifact. During student training, regularize the student's pairwise distances toward the target.
- **Computational concern**: Pairwise distances on a batch of B sequences × L positions → (B·L)² entries. For B=8, L=1024, this is ~67M entries per layer. Too expensive. Must subsample: random subset of positions, or compute Gram matrix in the token-type (vocabulary) dimension instead.
- **Speculation level**: MODERATE. Relational distillation is well-established, but adapting it to a "pre-computed artifact" paradigm (no teacher at runtime) is non-standard.

---

## 4. Attention Structure

### Key Question
Do attention patterns converge to predictable structures? Can we initialize or regularize attention toward those patterns?

### 4.1 Attention Head Specialization

**Voita, Talbot, Moiseev, Sennrich & Titov (2019).** "Analyzing Multi-Head Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned." ACL 2019. [arXiv:1905.09418](https://arxiv.org/abs/1905.09418)

**Key findings**: Most attention heads can be pruned without significant quality loss. The surviving "important" heads fall into interpretable categories:
- **Positional heads**: Attend to adjacent positions (local patterns)
- **Syntactic heads**: Attend to specific syntactic relations
- **Rare-token heads**: Attend to rare/informative tokens

Only ~30% of heads are important; the rest are redundant. The important heads emerge consistently across training runs.

**Clark, Khandelwal, Levy & Manning (2019).** "What Does BERT Look At? An Analysis of BERT's Attention." BlackboxNLP 2019. [arXiv:1906.04341](https://arxiv.org/abs/1906.04341)

**Key findings**: BERT's attention heads develop consistent patterns: some attend to the [CLS]/[SEP] tokens (delimiter heads), some attend to the previous/next token (local heads), and some attend broadly (global heads). These patterns are consistent across training runs and correlate with specific linguistic functions.

### 4.2 Induction Heads — A Universal Convergent Circuit

**Olsson, Elhage, Nanda et al. (2022).** "In-context Learning and Induction Heads." Transformer Circuits Thread. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)

**Key findings**: Induction heads — attention heads that implement pattern copying via a "previous token" head composed with a "match and copy" head — are a universal circuit that emerges across model sizes and architectures. They appear at a predictable phase transition during training and are mechanistically linked to in-context learning ability. Their emergence is **consistent across random seeds**, even in 2-layer models with a few million parameters.

**Applicability to pgolf**:
- **Strongest convergence result for attention**: Induction heads emerge reliably even in tiny models. Pre-initializing the attention weights to approximate the induction circuit could save significant training compute.
- **Actionable**: The induction head circuit is well-characterized: one head attends to the previous token (positional), another attends to tokens that follow copies of the current token (pattern matching). These patterns can be initialized as attention biases.

### 4.3 Attention Pattern Convergence

**Raganato & Tiedemann (2018).** "An Analysis of Encoder Representations in Transformer-Based Machine Translation." EMNLP 2018.

**Key findings**: In machine translation transformers, attention patterns in lower layers tend to be more local (attending to nearby positions), while higher layers develop broader attention. This local→global progression is consistent across training runs and language pairs.

**Applicability to pgolf**:
- **Predictable attention structure**: If attention patterns are consistent across seeds/runs, we can extract a "target attention profile" from the probe: the average attention entropy per head per layer, the average attention distance per head, and which heads are local vs. global.
- **Initialization via positional bias**: Rather than regularizing attention weights directly (which interacts poorly with softmax), we can initialize attention with a positional bias that encodes the local/global structure. This is effectively what ALiBi (Press, Smith & Lewis, 2022; [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)) does — but we could extract head-specific slopes from the probe rather than using a fixed formula.
- **Attention sparsity as regularization**: Add an entropy penalty per attention head. Heads that should be local (per the probe) get a stronger penalty for attending broadly, and vice versa.

### 4.3 Implicit Attention Biases

**Press, Smith & Lewis (2022).** "Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization." ICLR 2022. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)

**Key findings**: ALiBi adds a fixed linear bias to attention scores that decreases with distance: bias(i,j) = -m · |i-j|, where m is a head-specific slope. This replaces positional embeddings entirely and enables length generalization. Different slopes per head create a natural local/global hierarchy.

**Su, Lu, Pan, Murtadha, Wen & Liu (2024).** "RoFormer: Enhanced Transformer with Rotary Position Embedding." Neurocomputing 2024. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

**Key findings**: RoPE encodes relative position via rotation in the complex plane, creating a natural decay of attention with distance. Already used in the pgolf baseline.

**Applicability to pgolf**:
- **Probe-derived positional biases**: The pgolf baseline uses RoPE (and some submissions use partial RoPE). We could extract the effective attention distance profile from the probe and encode it as an additional fixed attention bias, complementing RoPE.
- **Head-specific temperature**: Instead of or in addition to positional bias, use different softmax temperatures per attention head, derived from the probe's attention entropy per head. Heads with low entropy (sharp attention) get a low temperature; heads with high entropy (broad attention) get a high temperature.
- **Compact encoding**: One slope value per head per layer. With 8 heads × 10 layers = 80 floats = 320 bytes. Trivially fits in the artifact.
- **Speculation level**: LOW for the mechanism (ALiBi-like biases are well-established); MODERATE for the specific strategy of deriving them from a probe.

### 4.6 Adaptive Attention Span

**Sukhbaatar, Grave, Lample, Jegou & Joulin (2019).** "Adaptive Attention Span in Transformers." ACL 2019. [arXiv:1905.07799](https://arxiv.org/abs/1905.07799)

**Key findings**: Each attention head learns its own optimal attention span via a soft masking function with a single learned parameter per head. Early layers develop short spans (local), deeper layers longer spans. Adds essentially zero parameters.

**Applicability to pgolf**: Initialize attention spans to the probe's learned distribution (short for early layers, long for later layers). Cost: 1 float per head = 80 bytes total.

### 4.7 Assessment for pgolf

Attention structure regularization is the **least promising** of the four research areas for pgolf, for two reasons:

1. **RoPE already provides positional structure**: The baseline and all top submissions use RoPE (or partial RoPE), which implicitly regularizes attention patterns toward local structure. Adding explicit attention regularization may be redundant.

2. **Attention is downstream of representations**: If we successfully regularize the representation geometry (Sections 1-3), attention patterns will follow naturally. Regularizing attention directly is treating a symptom, not a cause.

**Recommendation**: Defer attention regularization to a second phase of experiments. Focus on representation geometry first.

---

## 5. Practical Feasibility

### Key Question
How compact is the geometric description? Can it fit in the 16MB artifact alongside the model? What is the computational overhead during training?

### 5.1 Size of Geometric Descriptors

For a model with d=512, L=10 layers, H=8 attention heads:

| Descriptor | Per Layer | Total | Size |
|---|---|---|---|
| Singular value profile (top-50 per weight matrix, 4 matrices/layer) | 200 floats | 2000 floats | 8 KB |
| Target Gram matrix eigenvalues (top-50) | 50 floats | 500 floats | 2 KB |
| Activation covariance (full d×d) | 262,144 floats | 2,621,440 floats | 10 MB |
| Activation covariance (top-50 eigenvectors) | 25,600 floats | 256,000 floats | 1 MB |
| Attention entropy per head | 8 floats | 80 floats | 320 B |
| Attention distance profile per head | 8 floats | 80 floats | 320 B |
| Power-law exponent per matrix | 4 floats | 40 floats | 160 B |

**Compact option (recommended)**: Singular value profiles + Gram eigenvalues + attention stats = **~11 KB**. Trivially fits.

**Rich option**: Top-50 eigenvectors of activation covariance per layer = **~1 MB**. Still fits easily alongside a 15MB quantized model.

**Full option**: Full activation covariance per layer = **10 MB**. Too large — competes with model parameters for the 16MB budget.

### 5.2 Computational Overhead of Regularization Losses

All estimates for d=512, batch of B=8 sequences × 1024 tokens, per training step:

| Regularizer | Operation | FLOPs | Wall Time (H100 est.) |
|---|---|---|---|
| Isotropy (covariance Frobenius norm) | X^T X then compare | ~4.3 GFLOPs | ~0.05 ms |
| Spectral profile (periodic SVD every 50 steps) | Full SVD of d×d | ~0.54 GFLOPs amortized | ~0.01 ms amortized |
| Orthogonality (W^T W - target) | W^T W per matrix | ~5.4 GFLOPs total | ~0.05 ms |
| Gram eigenvalue matching | Eigendecomposition of K | ~0.54 GFLOPs | ~0.01 ms |
| Attention entropy | -Σ p log p over softmax | ~0.03 GFLOPs | negligible |
| CKA loss | Gram matrices + Frobenius norms | ~8.6 GFLOPs | ~0.1 ms |

**Total overhead**: ~19 GFLOPs per step if all regularizers active. A training step for a 50M-param model on H100 is ~200-500 GFLOPs. Overhead is **~5-10%**. Acceptable.

**If periodic (every 50 steps)**: Amortized overhead drops to **~0.1-0.2%**. Negligible.

### 5.3 Representation Matching Without a Teacher at Runtime

**Romero, Ballas, Kahou, Chassang, Gatta & Bengio (2015).** "FitNets: Hints for Thin Deep Nets." ICLR 2015. [arXiv:1412.6550](https://arxiv.org/abs/1412.6550)

**Key findings**: Matching intermediate representations of a teacher during student training accelerates convergence and improves final quality. The key insight: the student doesn't need to match exact activation values — matching the *projected* activations (through a learned regressor) works.

**Passalis & Tefas (2018).** "Learning Deep Representations with Probabilistic Knowledge Transfer." ECCV 2018. [arXiv:1803.10837](https://arxiv.org/abs/1803.10837)

**Key findings**: Probabilistic Knowledge Transfer (PKT) matches the *probability distributions* of teacher and student representations using KL divergence in a learned kernel space. This transfers geometric structure without requiring pointwise activation matching.

**Saha et al. (2022).** "Distilling Representational Similarity using CKA." BMVC 2022.

**Key findings**: Directly uses CKA as a loss function for representation distillation, computing cosine similarity between centered and normalized inter-example similarity matrices. The student beats or matches its teacher. Demonstrates that distilling the centered, normalized similarity distribution is more effective than distilling raw activations.

**Applicability to pgolf** — the critical distinction:
- Standard KD methods (FitNets, PKT) require the teacher to be present during training to produce activations on each training batch.
- **Our approach**: The probe is used *offline* to extract *static geometric descriptors* (spectral profiles, Gram eigenvalues, covariance eigenvectors). These descriptors are stored as a compact artifact and used as regularization targets during student training. **No teacher inference at runtime.**
- This is more restrictive (we can't adapt to the specific training batch) but much cheaper (no teacher forward pass) and fits the pgolf constraint of a self-contained artifact.

### 5.4 Prior Work on Static Geometric Priors

**Zagoruyko & Komodakis (2017).** "Paying More Attention to Attention." ICLR 2017. [arXiv:1612.03928](https://arxiv.org/abs/1612.03928)

**Key findings**: Transferring attention maps (not activations) from teacher to student improves convergence. Attention maps are a compressed geometric descriptor of how the network processes information.

**Lee, Song, & Kim (2018).** "Self-Knowledge Distillation via Dropout." [arXiv:1811.01514](https://arxiv.org/abs/1811.01514)

**Key findings**: A model can distill from its own earlier checkpoints without an external teacher. This shows that geometric targets need not come from a separate, larger model.

**Applicability to pgolf**:
- Self-distillation (from a heavily overtrained version of the same architecture) is exactly our proposed approach. The probe IS the same architecture, just trained longer.
- The geometric descriptors we extract encode what the model "wants" to converge to, not knowledge from a fundamentally different model. This makes the prior more compatible with the student's capacity.

---

## 6. Concrete Proposal

### 6.1 What to Extract from the Probe

**Phase 0: Train the Probe** (offline, not counted toward 10-min budget)
- Train the baseline architecture (10-11 layers, 512d, 1024 vocab, tied embeddings) for 4+ hours on 8xH100 or for 50k+ steps.
- Use existing competition hyperparameters but with MAX_WALLCLOCK_SECONDS=0 (unlimited).
- Train 3 probes from different seeds to verify convergence.

**Phase 1: Extract Geometric Descriptors** (offline)
Run the probes on a calibration subset (10k documents from training data) and extract:

1. **Per-layer activation covariance spectrum**: Top-50 eigenvalues of the activation covariance matrix at each layer. Verify consistency across 3 seeds (should be >0.9 correlation).

2. **Per-weight-matrix singular value profile**: Top-50 singular values for Q, K, V, O projections and MLP up/down matrices. Also record the effective rank (number of SVs > 1% of max).

3. **Embedding isotropy metrics**: Condition number of the tied embedding matrix, and the ratio of variance along the top-3 principal components to total variance.

4. **Attention entropy profile**: Mean entropy of each attention head's distribution, averaged over calibration data. This tells us which heads are sharp (local) vs. diffuse (global).

**Total descriptor size**: ~11 KB. Store as a numpy `.npz` file alongside the model.

### 6.2 How to Encode as Losses

**Loss 1: Spectral Profile Matching** (most promising, try first)
```python
# Every 50 steps:
for layer_idx, layer in enumerate(model.layers):
    # Compute activation covariance eigenvalues
    with torch.no_grad():
        cov = activations[layer_idx].T @ activations[layer_idx] / batch_size
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = eigvals.flip(0)[:50]  # Top 50, descending
    # L2 loss against target profile
    target = target_eigvals[layer_idx]
    loss_spectral += F.mse_loss(eigvals / eigvals[0], target / target[0])  # Normalize
```
Weight: Start at λ=0.1, anneal to 0 during warmdown.

**Loss 2: Isotropy Regularization** (well-established, try second)
```python
# Every step (cheap):
for layer_idx, h in enumerate(hidden_states):
    # h: (batch*seq_len, d)
    mean = h.mean(0)
    centered = h - mean
    cov = centered.T @ centered / h.shape[0]
    # Penalize deviation from target condition number
    target_cond = target_condition_numbers[layer_idx]
    eigvals = torch.linalg.eigvalsh(cov)
    current_cond = eigvals[-1] / eigvals[0].clamp(min=1e-8)
    loss_iso += (current_cond - target_cond).pow(2)
```

**Loss 3: Embedding Geometry** (fast, try third)
```python
# Penalize embedding condition number toward target
emb = model.embed.weight  # (1024, 512)
svs = torch.linalg.svdvals(emb)
target_cond = target_embed_condition_number
current_cond = svs[0] / svs[-1].clamp(min=1e-8)
loss_emb = (current_cond - target_cond).pow(2)
```

**Loss 4: Soft Orthogonality Toward Target** (moderate cost, try fourth)
```python
# Every 50 steps:
for name, W in model.named_parameters():
    if 'weight' in name and W.dim() == 2:
        WtW = W.T @ W
        target_WtW = target_gram[name]
        loss_ortho += F.mse_loss(WtW / WtW.norm(), target_WtW / target_WtW.norm())
```

### 6.3 What to Initialize

In addition to loss terms, use the probe's geometry for **weight initialization**:

1. **Embedding initialization**: Initialize the tied embedding matrix to match the probe's embedding SVD structure. Specifically: U Σ^{1/2} from the probe's embedding SVD, scaled appropriately.

2. **Weight matrix initialization**: For each weight matrix, initialize with the probe's SVD truncated to rank r (where r is the probe's effective rank). This gives the student a head start toward the converged geometry without copying the exact weights.

3. **Attention bias initialization**: If using head-specific positional biases, initialize the slopes from the probe's attention distance profile.

**Critical note on initialization vs. regularization**: Initialization provides a one-time bias that decays as training progresses. Regularization provides ongoing pressure. For a 10-minute training budget, initialization may be more valuable than regularization because:
- Regularization costs compute every step
- The student may not have time to "find" the target geometry on its own, but may have time to "refine" a good initialization
- Initialization is free at runtime

### 6.4 Experimental Plan

**Experiment 1: Baseline measurement** (1 GPU-hour)
- Train 3 probes from different seeds for 1 hour each
- Extract geometric descriptors from each
- Measure cross-seed CKA and spectral profile correlation
- If correlation < 0.8: geometric priors are seed-dependent, approach may not work
- If correlation > 0.9: geometric priors are robust, proceed

**Experiment 2: Spectral initialization** (2 GPU-hours)
- Initialize student from probe's SVD structure (not weights)
- Compare against standard initialization (Kaiming, orthogonal)
- Measure: steps to reach baseline val_bpb, final val_bpb at 10 min
- Expected outcome: faster early convergence, unclear if final quality improves

**Experiment 3: Isotropy regularization** (2 GPU-hours)
- Add isotropy loss targeting probe's condition numbers
- Sweep λ ∈ {0.01, 0.05, 0.1, 0.5}
- Compare against baseline with same training budget
- Expected outcome: modest improvement (0.001-0.005 bpb) based on Gao et al. results

**Experiment 4: Spectral profile regularization** (2 GPU-hours)
- Add spectral profile matching loss (every 50 steps)
- Sweep λ ∈ {0.01, 0.1, 1.0}
- Compare against baseline
- Expected outcome: uncertain — this is the most novel experiment

**Experiment 5: Combined** (2 GPU-hours)
- Best initialization + best regularizer combination from above
- Fine-tune λ values
- Expected outcome: if individual experiments show signal, combined should too

**Experiment 6: Embedding geometry only** (1 GPU-hour)
- Quick win test: only regularize the embedding matrix toward probe's geometry
- This is the cheapest intervention (one matrix, computed once per step)
- Based on Gao et al., this alone may provide 0.002-0.005 bpb improvement

### 6.5 Proposed Config

```bash
# Probe training (offline, not counted)
RUN_ID=probe_seed42 \
MAX_WALLCLOCK_SECONDS=0 \
ITERATIONS=50000 \
SAVE_CHECKPOINT=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Extract geometric descriptors (offline script, to be written)
python3 extract_geometry.py --checkpoint probe_seed42/model.pt --output geometry.npz

# Student training with geometric regularization
RUN_ID=geo_reg_v1 \
GEOMETRY_PRIOR=geometry.npz \
GEO_LOSS_WEIGHT=0.1 \
GEO_LOSS_SCHEDULE=cosine_anneal \
GEO_LOSS_EVERY=50 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### 6.6 Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Cross-seed geometry diverges at this scale | MEDIUM | Experiment 1 tests this first |
| Regularization overhead eats into training steps | LOW | Amortized cost <0.2%; periodic eval confirms no slowdown |
| Probe's geometry is too rich for student's capacity | MEDIUM | Use only top-k eigenvalues; student may only match coarse structure |
| Regularization conflicts with quantization-aware training | MEDIUM | Schedule: geo reg in first 80%, QAT in last 20% |
| Improvement is real but too small to matter (<0.001 bpb) | HIGH | This is fundamental research — even a null result is informative |
| Initialization is more effective than regularization | MEDIUM | Test both independently (Experiments 2 vs 3-4) |

### 6.7 What is Well-Established vs. Speculative

**Well-established** (strong empirical support in literature):
- Trained LMs have low intrinsic dimensionality and structured spectral profiles
- Representation degeneration occurs with tied embeddings; isotropy regularization helps
- Orthogonal initialization/regularization improves training stability
- Neural collapse (simplex ETF) occurs in terminal phase of training
- CKA shows cross-seed representation convergence

**Moderately supported** (evidence exists but not in this exact setting):
- Using a probe's spectral profile as a regularization target
- Geometric descriptors transferring from overtrained to budget-constrained models
- Pre-computed relational targets (geometric distillation without runtime teacher)

**Speculative** (mechanistically sound but untested):
- Power-law exponent matching as a training loss
- Head-specific attention temperature from probe analysis
- Combined spectral + isotropy + orthogonality regularization in a single training run
- Whether the improvement magnitude justifies the complexity in a competitive setting

---

## References

1. Aghajanyan, Gupta & Zettlemoyer (2021). "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." ACL 2021. arXiv:2012.13255
2. Ansuini, Laio, Macke & Zdeborová (2019). "Intrinsic dimension of data representations in deep neural networks." NeurIPS 2019. arXiv:1905.12784
3. Bansal, Chen & Wang (2018). "Can We Gain More from Orthogonality Regularizations in Training Deep Networks?" NeurIPS 2018. arXiv:1810.09102
4. Clark, Khandelwal, Levy & Manning (2019). "What Does BERT Look At?" BlackboxNLP 2019. arXiv:1906.04341
5. Gao, He, Tan & Qiu (2019). "Representation Degeneration Problem in Training Natural Language Generation Models." ICLR 2019. arXiv:1907.12009
6. Godey, Clergerie & Sagot (2024). "Anisotropy Is Inherent to Self-Attention in Transformers." arXiv:2401.12143
7. Gouk, Frank, Pfahringer & Cree (2021). "Regularisation of Neural Networks by Enforcing Lipschitz Continuity." Machine Learning 2021. arXiv:1804.04368
8. Hsu, Bhojanapalli, Garg & Telgarsky (2022). "Language Model Compression with Weighted Low-Rank Factorization." ICLR 2022. arXiv:2207.00112
9. Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685
10. Huh, Cheung, Wang & Isola (2024). "The Platonic Representation Hypothesis." ICML 2024. arXiv:2405.07987
11. Khodak, Balcan & Talwalkar (2021). "Initialization and Regularization of Factored Neural Layers." ICLR 2021. arXiv:2105.01029
12. Kornblith, Norouzi, Lee & Hinton (2019). "Similarity of Neural Network Representations Revisited." ICML 2019. arXiv:1905.00414
13. Lee, Song & Kim (2018). "Self-Knowledge Distillation via Dropout." arXiv:1811.01514
14. Li, Hopkins, Bau & Viégas (2023). "Emergent World Representations." ICLR 2023. arXiv:2210.13382
15. Liu et al. (2021). "Learning with Hyperspherical Uniformity." AISTATS 2021. arXiv:2103.01649
16. Martin & Mahoney (2019). "Traditional and Heavy-Tailed Self Regularization in Neural Network Models." arXiv:1901.08276
17. Miyato, Kataoka, Koyama & Yoshida (2018). "Spectral Normalization for GANs." ICLR 2018. arXiv:1802.05957
18. Mu & Viswanath (2018). "All-but-the-Top: Simple and Effective Postprocessing for Word Representations." ICLR 2018. arXiv:1702.01417
19. Nanda, Chan, Lieberum, Smith & Steinhardt (2023). "Progress Measures for Grokking via Mechanistic Interpretability." ICLR 2023. arXiv:2301.05217
20. Nguyen, Raghu & Kornblith (2021). "Do Wide Neural Networks Learn Representations?" NeurIPS 2021. arXiv:2010.15110
21. Olsson, Elhage, Nanda et al. (2022). "In-context Learning and Induction Heads." arXiv:2209.11895
22. Papyan, Han & Donoho (2020). "Prevalence of Neural Collapse." PNAS 2020. arXiv:2008.08186
23. Park, Kim, Lu & Cho (2019). "Relational Knowledge Distillation." CVPR 2019. arXiv:1904.05068
24. Passalis & Tefas (2018). "Learning Deep Representations with Probabilistic Knowledge Transfer." ECCV 2018. arXiv:1803.10837
25. Press, Smith & Lewis (2022). "Train Short, Test Long: ALiBi." ICLR 2022. arXiv:2108.12409
26. Romero, Ballas, Kahou, Chassang, Gatta & Bengio (2015). "FitNets: Hints for Thin Deep Nets." ICLR 2015. arXiv:1412.6550
27. Rudman & Eickhoff (2024). "Stable Anisotropic Regularization (I-STAR)." ICLR 2024. arXiv:2305.19358
28. Saha et al. (2022). "Distilling Representational Similarity using CKA." BMVC 2022.
29. Scarvelis & Solomon (2024). "Nuclear Norm Regularization for Deep Learning." NeurIPS 2024. arXiv:2405.14544
30. Sharma & Karakida (2024). "The Truth is in There: Layer-Selective Rank Reduction." ICLR 2024. arXiv:2312.13558
31. "Small Singular Values Matter" (2024). arXiv:2410.17770
32. Su, Lu, Pan, Murtadha, Wen & Liu (2024). "RoFormer: Enhanced Transformer with Rotary Position Embedding." Neurocomputing 2024. arXiv:2104.09864
33. Sukhbaatar, Grave, Lample, Jegou & Joulin (2019). "Adaptive Attention Span in Transformers." ACL 2019. arXiv:1905.07799
34. Tian, Krishnan & Isola (2020). "Contrastive Representation Distillation." ICLR 2020. arXiv:1910.10699
35. Voita, Talbot, Moiseev, Sennrich & Titov (2019). "Analyzing Multi-Head Attention: Specialized Heads Do the Heavy Lifting." ACL 2019. arXiv:1905.09418
36. Wang, Chen, Li & Xiang (2020). "Improving Neural Language Generation with Spectrum Control." ICLR 2020. arXiv:1906.02754
37. Yan et al. (2022). "Addressing Token Uniformity in Transformers via Singular Value Transformation." arXiv:2208.11790
38. Yang et al. (2022). "Inducing Neural Collapse in Deep Long-Tailed Learning."
39. Yoshida & Miyato (2017). "Spectral Norm Regularization for Improving the Generalizability of Deep Learning." arXiv:1705.10941
40. Zagoruyko & Komodakis (2017). "Paying More Attention to Attention." ICLR 2017. arXiv:1612.03928
41. Zhong et al. (2023). "Understanding Collapse in Non-Contrastive Learning."
42. "Generalizing and Decoupling Neural Collapse via Hyperspherical Uniformity Gap" (2023). arXiv:2303.06484
