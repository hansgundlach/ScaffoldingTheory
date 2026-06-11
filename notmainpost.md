<!-- 

# Just a Wrapper? <!-- This is for a technical ish blog post on FutureTech's substack Mixture of Experts. Similar level to some of Epoch's Gradient updates --> <!-- Behind Every Great Model is a great scaffold --> <!-- If Claude has seen futher, it has been on the sholder of scaffolds -->' <!--

pasted

Can you improve the writing here PLEASE RETURN AS MARKDOWN CODE CODE CODE CODE COCE ? (excpet in under development section leave htis alone):

Jun 7

Refined blog post prose while preserving development section untouched
Refined blog post prose while preserving development section untouched
Here's the cleaned-up version. I fixed typos, tightened the prose, repaired the broken sentence in the intro, made "Specialist Scaffolding..." a proper header, removed em dashes (per your usual style), and left the "(still Developing)" block completely untouched.

markdown
# Just a Wrapper?

<!-- This is for a technical ish blog post on FutureTech's substack Mixture of Experts. Similar level to some of Epoch's Gradient updates -->

<!-- Behind Every Great Model is a great scaffold -->
<!-- If Claude has seen further, it has been on the shoulder of scaffolds -->
<!-- Ask not what your scaffold can do for your model but what your model can do for your scaffold -->

I am the shape the water takes.

- ClawdBot, [Moltbook](https://www.astralcodexten.com/p/best-of-moltbook?hide_intro_popup=true)


**TL;DR:**
- Specialist scaffolding improvements can be very significant, representing as much as two years' worth of algorithmic progress in a given domain.
- The effect of scaffolding is model-dependent: a given scaffold helps some models while hindering others. Scaffolding therefore represents a unique case of AI enhancement and algorithmic progress.
- These scaffold-model interactions have important implications for evaluating, ranking, and comparing models, and for understanding the agentic economy.
- In particular, model creators have a unique advantage in improving the downstream performance of agentic systems built on their own models.

When we think about AI improvement, we usually focus on pretraining, reinforcement learning, and inference efficiency. But scaffolding, meaning everything else around the model, is often overlooked despite playing an important role. If we want a full picture of what drives progress in language models, we have to start accounting for scaffolding. Here we begin to get a handle on these effects using data from the Holistic Agent Leaderboard (HAL).

The Holistic Agent Leaderboard is an index of model performance on agentic benchmarks that records both accuracy and price. Importantly, it runs each model on multiple scaffolds (two to three) for every benchmark. Most benchmarks are paired with a scaffold specific to that benchmark. On CORE-bench, for example, models are run with the CORE-Agent scaffold alongside the HAL generalist scaffold. The HAL generalist agent is built on the smolagents framework, in which the model performs all of its actions by writing Python code. To search the internet, for instance, the agent writes Python code that performs the search. HAL covers AssistantBench, CORE-Bench Hard, GAIA, Online Mind2Web, SWE-bench Verified Mini, SciCode, ScienceAgentBench, TAU-bench Airline, and USACO.

# What is Scaffolding?

Before going deeper, we should clarify what scaffolding is. Scaffolding is everything that is not included in traditional AI scaling. It can include:

- Tool affordances: bash terminal access, file editing tools
- Context window management: when retrieval is used, how compaction is done
- Memory and summarization
- Prompting: system prompts, submission instructions, planning prompts
- Reflection and self-critique steps
- Token, time, and cost limits, and stopping rules
- The agent loop: think → act → observe → repeat

In this post, we consider all of these in aggregate. Future work would ideally disentangle the relative importance of each component. For example, are scaffolds mostly the sum of their prompts, or do they depend more on heavier engineering?

# Specialist Scaffolding Has a Large Impact on AI Performance

Given the protean nature of scaffolding, it is not surprising that its effects are large and vary widely. A single scaffold can significantly improve one model's performance while hurting another's. To examine this, we plot each model's final accuracy against the API cost needed to reach it, applying a logit transform to the accuracy values across all scaffolds.

Below are per-scaffold price-performance frontiers for two benchmarks (CORE-bench Hard and SciCode), showing how accuracy and cost vary across scaffolds for each model. Scaffolding clearly has a large impact on performance.[^1] Switching from a generalist scaffold to a specialist one can deliver the same accuracy at 100x lower cost. For context, algorithmic progress in AI inference typically lowers prices by about 10x per year at a fixed level of performance (see our [paper](https://arxiv.org/pdf/2511.23455)). This means that switching to a specialist scaffold can, in some cases, be like jumping to an AI model two years in the future. That said, some scaffolds have no clear effect on the Pareto frontier. Finally, the Claude Code scaffold paired with Claude models produces exceptionally high performance. We investigate why later in the piece.

[^1]: Although we do not have precise estimates of run-to-run variability from HAL, benchmark variation is generally quite small for most tasks. On Epoch's benchmark hub, for instance, the standard error between runs is usually only 1–2%.

The effect of scaffolding can be large enough that in some cases it makes more sense to benchmark scaffolds than models, meaning a leaderboard of scaffolds would be more informative than a leaderboard of models. In the appendix, we include a variance decomposition that quantifies how much performance variation is due to the model versus the scaffold. For some benchmarks, almost as much variation is due to scaffolding as to the model. To put this in context, it is almost as if the model is more of a utility for the scaffold than the other way around.

<table>
<tr>
<td><img src="figures/hal_frontier_core_bench_hard.png" width="100%"></td>
<td><img src="figures/hal_frontier_scicode.png" width="100%"></td>
</tr>
</table>

<sub><i>
<b>Left:</b> CORE-bench Hard. Clear and substantial differences between scaffolds in cost versus accuracy, with improvements sometimes spanning two orders of magnitude.<br>
<b>Right:</b> SciCode. Scaffold differences on the Pareto frontier are irregular but still noticeable.<br>
<b>Orange points:</b> Claude models using the Claude Code scaffold, which show exceptionally high performance.
</i></sub>


The scaffold-switch vectors below trace the same model moving between two scaffolds. Each arrow points from one scaffold's (cost, accuracy) to another's, colored by whether the switch made the model more or less accurate and more or less expensive.

<table><tr>
<td><img src="figures/hal_vectors_gaia__hal_generalist_agent_vs_hf_open_deep_research.png" width="100%"></td>
<td><img src="figures/hal_vectors_sciagentbench__hal_generalist_agent_vs_sab_self_debug.png" width="100%"></td>
</tr></table>

<sub><i>
<b>Left:</b> Switching scaffolds on the GAIA benchmark produces highly varied effects. Some models become both cheaper and more accurate, while others see reduced accuracy, increased cost, or both.<br>
<b>Right:</b> On SciAgentBench, changing scaffolds consistently makes models both more accurate and less expensive, showing a uniform benefit.
</i></sub>


Here we see that scaffolding has significant interaction effects, particularly on the GAIA benchmark. Switching to the HF Open Deep Research scaffold significantly improves some models while hurting others. This contrasts with most algorithmic improvements in AI, which act as rising tides that lift all boats. The invention of the GeLU activation function or flash attention, for instance, **benefits every model**. The nonuniform effect of scaffolding has several important implications. The first is that most model leaderboards are misleading.

# Scaffolding Effect on Ranking

If scaffolds have nonuniform effects, then a model that ranks 20th under one scaffold could rank 3rd under another. Model leaderboards aim to capture either the performance of models in deployment or their maximum capabilities. To do this well, it may be necessary to run each model across several scaffolds and take the maximum performance, or to determine which scaffold best matches actual deployment. Going further, the effect of scaffolding can be large enough that evaluators should arguably build leaderboards of scaffolds rather than of models. We include our formal analysis of scaffold effects on benchmark ranking in the appendix. For some benchmarks, such as SciCode, rankings are well preserved across scaffolds. For most, though, measures of ranking preservation indicate large changes.

<!-- 
- Since Scaffolding has interactions model ranking is not useful 
- model Evaluators ideally should test models on multiple scaffolds and take the maximum to get a better ranking  -->

=======================================

(still Developing)
# Claude Scaffold is Suspiciously Strong

# Why This Can Change The Agent Economy

- vertical integration

Claude on Claude Code Harness Performance vs on regular harness
Random Modles on Claude Code Harness vs on regular hanress

- Algorithmic Progress Only Available to Model Creators

- Are Scaffolds Democratizing? 
- low compute way to drasticaly increase model capabilites 
- larger model creaters have a unique cointegration benefits. 

4.6 Sonnet

# Conclusion and Future Work 
- scaffolding point to broader interoprability concerns
- scafolding research should try to decompose these effects if possible 
-Empricical work to identify the extent of cooptimization. 


===========================================

# Appendix

## Shapley-Value Decomposition Analysis

We apply a Shapley-value decomposition to separate the effects of models and their scaffolds. Models explain the majority of variance across all benchmarks we measure. For some benchmarks, however, the effect of scaffolding rivals that of the model itself: the variance in performance is almost as well accounted for by variation in scaffolding as by variation in the model. We also believe these estimates understate the importance of scaffolding, since a linear decomposition does not capture interaction effects between models and scaffolds.

<!-- 
To quantify how much of the spread in agent accuracy comes from the *model* versus the *scaffold*, we fit OLS models on logit-transformed accuracy for each benchmark and use a **Shapley-value decomposition** of the regression R² (the LMG / Shapley-regression method). Each factor's contribution is the average of its marginal increase in R² over every ordering in which factors are added — for two factors, `Shapley(model) = ½·R²(model) + ½·[R²(both) − R²(scaffold)]` and symmetrically for the scaffold. Unlike Type-II sum-of-squares, the Shapley shares are order-independent and sum exactly to the full-model R², so the residual is simply `1 − R²(both)`. The model term is generally the larger driver, but on SciAgentBench, TAU-bench Airline, and SWE-bench Mini Verified the scaffold accounts for roughly a third of explained variance — nearly as much as the model itself. This is the formal version of the claim that, on some benchmarks, the choice of scaffold matters almost as much as the choice of model. -->

<p align="center"><img src="figures/anova_variance_decomposition.png" width="520"></p>


## Full Pareto Analysis Graphs

Per-scaffold price-performance frontiers across all benchmarks. Each panel plots final accuracy (logit-scaled) against the API cost needed to reach it, with one frontier per scaffold.

<p align="center"><img src="figures/hal_price_performance_frontier.png" width="900"></p>

## Full Vector Analysis Graphs

Scaffold-switch vectors for the remaining benchmark and scaffold pairs. Each arrow traces a single model moving between two scaffolds, from one scaffold's (cost, accuracy) to the other's, colored by whether the switch made the model more or less accurate and more or less expensive, as in the figures in the main text. As before, we see two types of scaffolding change: rising-tide changes and turbulent changes. We observe roughly equal numbers of each.

<table>
<tr>
<td><img src="figures/hal_vectors_core_bench_hard__core_agent_vs_claude_code.png" width="440"></td>
<td><img src="figures/hal_vectors_core_bench_hard__core_agent_vs_hal_generalist_agent.png" width="440"></td>
</tr>
<tr>
<td><img src="figures/hal_vectors_core_bench_hard__claude_code_vs_hal_generalist_agent.png" width="440"></td>
<td><img src="figures/hal_vectors_swe_bench_mini_verified__hal_generalist_agent_vs_swe_agent.png" width="440"></td>
</tr>
<tr>
<td><img src="figures/hal_vectors_tau_bench_airline__hal_generalist_agent_vs_tau_bench_tool_calling.png" width="440"></td>
<td><img src="figures/hal_vectors_online_mind2web__browser_use_vs_seeact.png" width="440"></td>
</tr>
<tr>
<td><img src="figures/hal_vectors_scicode__hal_generalist_agent_vs_scicode_tool_calling_agent.png" width="440"></td>
<td><img src="figures/hal_vectors_scicode__hal_generalist_agent_vs_scicode_zero_shot_agent.png" width="440"></td>
</tr>
<tr>
<td><img src="figures/hal_vectors_scicode__scicode_tool_calling_agent_vs_scicode_zero_shot_agent.png" width="440"></td>
<td><img src="figures/hal_vectors_usaco__hal_generalist_agent_vs_usaco_episodic_semantic.png" width="440"></td>
</tr>
</table>

## Ranking Correlation Graphs

How well does a model's ranking under one scaffold predict its ranking under another? The chart below shows Spearman ρ and Kendall τ rank correlations between scaffold pairs for each benchmark. Most pairs preserve ranking moderately well, but CORE-bench Hard (CORE-Agent to Claude Code) is notably *negative*: the best models under one scaffold are not the best under the other.

![Rank-order preservation across scaffolds](figures/rank_correlation_summary.png)


<!-- ==============================================
Junk :
= -->

<!-- System prompt
Developer prompt
Task instructions
Tool definitions
Tool-use format
Bash/terminal access
File editing tools
Code search tools
Test-running tools
Docker/sandbox environment
Repo setup and dependencies
Agent loop: think → act → observe → repeat
Observation formatting
Context-window management
Memory/summarization
Retrieval or repo-map system
Planning prompts
Reflection/self-critique steps
Multi-agent reviewer/coder setup
Stopping rules
Token/time/cost limits
Patch submission format
Post-processing of diffs
Evaluation harness
Caching strategy -->
-
<!-- 

# Does Scaffolding Matter Less At Higher Capabilities

- graphs of capabilitey correlation --></document_content>
Two things worth flagging:

Method-name mismatch. The main text originally called the appendix analysis an "ANOVA," but the appendix performs a Shapley-value decomposition (and the image is anova_variance_decomposition.png). I changed the main-text reference to the neutral "variance decomposition" so it doesn't contradict the appendix, but you may want to rename the figure file or pick one term throughout.
The 100x claim. The body says switching scaffolds gives "100x cheaper performance" while the TL;DR and caption say "two orders of magnitude" / "two years." These line up at 10x/year, so they're consistent, but you assert "as much as two years" in the TL;DR and "can in some cases be like" in the body. Worth making sure the strongest framing matches what the CORE-bench figure actually shows.


Claude works directly with your codebase

Let Claude edit files, run commands, and ship changes from the desktop app, your terminal, or your IDE.

how i 



Claude is AI and can make mistakes. Please double-check responses. -->
