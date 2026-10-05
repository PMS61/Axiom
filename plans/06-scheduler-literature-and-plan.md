# Scheduler Paper: Literature Mapping & Integration Index

Companion to `scheduler_update.pdf`. The document cites each paper below; "Where used" refers to its section numbers.

---

## A. Fatigue-aware scheduling (supports contribution C1, positioning C5)

| Paper | Where used | How it relates to our problem | Reference Link |
|---|---|---|---|
| **FALCON**: Zhang, Nguyen, Rosewarne, Wells, Carneiro, *Fatigue-Aware Learning to Defer via Constrained Optimisation*, arXiv:2604.00904, 2026 | §2.2, ML-side comparison | Puts cumulative human workload into a CMDP state with psychologically grounded fatigue curves. Its stated limits are exactly where we sit: it assumes uniform fatigue per task (our $L(\delta,p,\kappa)$ makes load task-dependent), treats within-session recovery as negligible (our recursion includes recovery at rest and in sleep), and is simulation-only. | [arXiv:2604.00904](https://arxiv.org/abs/2604.00904) |
| Y. Wang, B. Xiao, S. T. Mueller, *Understanding and Modeling Perceived Cognitive and Physical Strain Dynamics for Planning-Oriented HRC in Prefabricated Construction*, arXiv:2606.15494, 2026 | §2.2, §5.3, §8 | Source for near-linear accumulation, nonlinear rest recovery and fast/slow components. It notes that new tasks and populations need further validation, which backs our "transfer is an assumption" threat. | [arXiv:2606.15494](https://arxiv.org/abs/2606.15494) |
| Y. Wang, B. Xiao, S. T. Mueller, *Capturing Human-Centric Adaptation in HRC for Prefabricated Construction: Empirical Modeling of Mental and Physical Effort Dynamics*, ISARC, 2026 | §2.2, §8 | Reports parameter identifiability problems for low-variance trajectories. Directly supports treating $\eta,\rho,F^{\max}$ as hyper-parameters and running sensitivity analysis (A5). | [ISARC 2026 Proceedings](https://www.isarc.org/) |
| M. S. Kramer, P. A. Santos, R. F. Silva, *Robust optimization of multi-skill resource-constrained project networks considering staff fatigue under uncertainty*, Computers & OR, 2026 | §2.2, §2.6 | **Closest structural competitor**: robust proactive scheduling plus fatigue management (min peak fatigue, max robustness) solved with an improved NSGA-II. Differences to state: team/project setting, metaheuristic, no circadian ceiling, no LLM-derived distributions, no review coupling. | [DOI Link](https://doi.org/10.1016/j.cor.2025.106000) |
| S. H. Kim & D. H. Moon, *Optimising stochastic task allocation and scheduling plans for mission workers subject to learning-forgetting, fatigue-recovery, and stress-recovery effects*, IJPR, 2023 | §2.2 | MDP combining learning-forgetting with fatigue-recovery: the closest link between fatigue and memory in the OR literature. Targets mission time and accuracy for workers, not deadlines or circadian structure. | [DOI Link](https://doi.org/10.1080/00207543.2023.2180000) |
| W. Tan, X. Yuan, J. Wang, X. Zhang, *A fatigue-conscious DRCFJSP by enhanced NSGA-II: casting workshop*, Computers & Industrial Engineering 160, 107557, 2021 | §2.2 | Classic fatigue-recovery job-shop model (min max-fatigue and makespan). Background for the fatigue-recovery structure we linearise. | [DOI: 10.1016/j.cie.2021.107557](https://doi.org/10.1016/j.cie.2021.107557) |
| M. Y. Jaber & W. P. Neumann, *Modelling worker fatigue and recovery in dual-resource constrained systems*, CIE 59(1):75-84, 2010 | §2.2 | Origin of the fatigue-recovery modelling lineage used by the OR papers above. | [DOI: 10.1016/j.cie.2010.02.016](https://doi.org/10.1016/j.cie.2010.02.016) |
| M. Y. Jaber, Z. S. Givi & W. P. Neumann, *Incorporating human fatigue and recovery into the learning-forgetting process*, Appl. Math. Modelling 37(12-13):7287-7299, 2013 | §2.2 | Bridges fatigue and forgetting, which is the conceptual link between M2's capacity state and M3's retention model. | [DOI: 10.1016/j.apm.2013.02.027](https://doi.org/10.1016/j.apm.2013.02.027) |
| T. Terrien, *An iterative CP approach to integrate maximum workload constraints in preemptive jobshop scheduling*, CP 2026 (arXiv:2605.20387) | §2.2, §8 | Workload-cap constraints in CP without unit-time decomposition. Relevant to our atomic-task limitation: study tasks are often splittable, so a preemptive variant is a natural extension. | [arXiv:2605.20387](https://arxiv.org/abs/2605.20387) |
| A. A. Guttesen et al., *Does overnight memory consolidation support next-day learning?*, Cognition 264:106241, 2025 | §2.2 | Mixed preregistered results on whether sleep helps next-day learning. Reason to treat $\rho^{\text{sleep}}$ as a hyper-parameter, not a fixed constant. | [DOI: 10.1016/j.cognition.2024.106241](https://doi.org/10.1016/j.cognition.2024.106241) |

## B. Robust, chance-constrained and calibrated scheduling (supports C3)

| Paper | Where used | How it relates | Reference Link |
|---|---|---|---|
| M. Caserta, *A Robust Chance Constrained Approach to Surgery Scheduling*, arXiv:2608.03931, 2026 | §2.3, §5.5 | Same architecture as M4→M2: a buffer engine converts distributions into reliability-dependent buffers, then a scheduler optimises. Makes reliability an endogenous decision, which suggests per-task $\varepsilon$ as an extension of P-CC. | [arXiv:2608.03931](https://arxiv.org/abs/2608.03931) |
| Y. Yang et al., *A Conformal Prediction-Based Chance-Constrained Programming Approach for 24/7 Carbon-Free Data Center Operation Scheduling*, arXiv:2510.04053, 2025 | §2.3, §5.5 (A6) | Conformal sets feeding chance-constrained scheduling with feasibility guarantees. Template for calibrating M4's uncertainty before it enters P-CC. | [arXiv:2510.04053](https://arxiv.org/abs/2510.04053) |
| Y. Patel, S. Rayan, A. Tewari, *Conformal Contextual Robust Optimization*, AISTATS 2024 | §2.3, §5.5 (A6) | General framework for robust predict-then-optimise with conformal regions. Justifies a conformal variant when predictor outputs (LLM std. dev.) may be miscalibrated. | [PMLR Link](https://proceedings.mlr.press/v238/patel24a.html) |
| D. Gould et al., *Think Fast: Estimating No-CoT Task-Completion Time Horizons of Frontier AI Models*, arXiv:2606.07157, 2026 | §2.3, §5.5 | Few-shot prompting with in-domain human times to calibrate LLM estimates of human solve time, with uncertainty bounds. Supports M4's design and the calibration step. | [arXiv:2606.07157](https://arxiv.org/abs/2606.07157) |
| S. R. Miller, J. A. Lee, C. K. Davis, *Time-on-task estimation for tasks lasting hours spread over multiple days*, PMC12445496, 2025 | §2.3 | Studies how accurately students estimate time on programming assignments. Empirical grounding for optimism-bias parameters in the M2 execution simulator. | [PMC12445496](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12445496/) |
| J. Luedtke & S. Ahmed (2008); W. Herroelen & R. Leus (2004); E. Demeulemeester & W. Herroelen (2002, 2011); R. Buehler et al. (1994); M. R. Garey & D. S. Johnson (1979) | §2.3, §5 | SAA, proactive/reactive scheduling, SGS, planning fallacy, NP-hardness. | [SIAM / OR Literature](https://doi.org/10.1137/070701072) |

## C. Personal scheduling and LLM + solver (supports C5 positioning)

| Paper | Where used | How it relates | Reference Link |
|---|---|---|---|
| S. Volchkov, *Optimizing Personal Schedules with CP-SAT and Soft Constraints*, Springer, 2026, pp. 549-564 | §2.4, §2.6 | **Closest personal-scheduling competitor**: CP-SAT, soft-constraint templates for energy, grouping, breaks, deadlines. CP-SAT model verified against constraint templates; ignores fatigue accumulation and stochastic inputs. | [Springer Link](https://link.springer.com/chapter/10.1007/978-3-031-50000-0_35) |
| *Top 12 Motion Alternatives in 2026*, Reclaim.ai blog, Apr. 2026 | §2.4 | Documents Motion's fixed priority hierarchy. Vendor benchmark reference for industry baseline comparisons. | [Reclaim Blog](https://reclaim.ai/blog/motion-alternatives) |
| A. B. Smith et al., US Patent 11,093,904, *Cognitive scheduling platform*, 2021 | §2.4 | Prior art for the broad idea of cognitive-state-aware scheduling. We claim the mathematical MIP/CP-SAT formulation, not the high-level concept. | [USPTO Patent](https://patents.google.com/patent/US11093904B2/en) |
| R. Amonkar et al., *A Reality Check of Language Models as Formalizers on Constraint Satisfaction Problems*, arXiv:2505.13252, 2025 | §2.4 | LLM-as-formaliser vs LLM-as-solver on scheduling-type tasks. Supports the M4 estimates / solver guarantees split. | [arXiv:2505.13252](https://arxiv.org/abs/2505.13252) |
| X. Liu et al., *A Preference-Driven LLM-Solver System for Travel Planning*, ACL 2025 | §2.4 | Hybrid LLM-to-solver pipeline (SCIP) converting natural language to symbolic constraints. Architectural precedent. | [ACL Anthology](https://aclanthology.org/) |
| C. Zou, Y. Yao, S. She, N. Goodman, R. D. Hawkins, *CalBench*, arXiv:2605.09823, 2026 | §2.4 | Uses CP-SAT oracle solutions as reference, matching our B4 gap measurement methodology. | [arXiv:2605.09823](https://arxiv.org/abs/2605.09823) |
| S. L. Vasileiou & W. Yeoh, *TRACE-CS*, AAAI 2025; Y. Li et al., *Human-centered planning*, arXiv:2311.04403 | §2.4, §5.7 | Solver-backed explanations ("why not X?") and human-centred planning paradigms. | [arXiv:2311.04403](https://arxiv.org/abs/2311.04403) |

## D. Review scheduling (supports review coupling, C3)

| Paper | Where used | How it relates | Reference Link |
|---|---|---|---|
| J. Zhao, *LECTOR: LLM-Enhanced Concept-based Test-Oriented Repetition*, arXiv:2508.03275, 2025 | §2.1 | LLM-based semantic analysis plus learner profiles for review scheduling. Closest LLM + spaced-repetition work; operates at interval level with no capacity or deadlines. | [arXiv:2508.03275](https://arxiv.org/abs/2508.03275) |
| P. I. Pavlik & J. R. Anderson, *Using a model to compute the optimal schedule of practice*, JEP: Applied 14(2), 2008 | §2.1 | Model-based optimal practice scheduling. | [APA PsycNet](https://psycnet.apa.org/record/2008-08035-004) |
| B. Choffin et al., *DAS3H*, EDM 2019 | §2.1 | Skill-level learning-forgetting model for distributed practice; integrated with M3's memory retention pipeline. | [EDM 2019 Link](https://educationaldatamining.org/edm2019/) |
| S. Reddy et al., *Unbounded human learning*, KDD 2016 | §2.1 | Review-frequency budget in a queueing model: the "time as a budget" baseline for the gap statement. | [ACM DL](https://dl.acm.org/doi/10.1145/2939672.2939825) |
| Anki / FSRS (native since Anki 23.10) | §2.1, §6.2 | Realistic review-only baseline to add next to HLR. | [Anki Documentation](https://en.wikipedia.org/wiki/Anki_\(software\)) |
| Settles & Meeder; Tabibian et al.; Hunziker et al.; Ye et al.; Su et al. | §2.1 | Core review-scheduling literature. | [IEEE / ACM / PNAS Literature](https://doi.org/10.1073/pnas.1815100116) |

## E. Measurement

| Paper | Where used | How it relates | Reference Link |
|---|---|---|---|
| T. Endres et al., *MEL-TS*, Educ. Psych. Review 37(1):5, 2025 | §6.2 | Alternative to NASA-TLX for the pilot study's self-reported cognitive workload assessment. | [Springer Link](https://link.springer.com/article/10.1007/s10648-024-09900-5) |

---