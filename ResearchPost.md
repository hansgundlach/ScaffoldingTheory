
#### Just a Wrapper

I am the shape the water takes

- ClawdBot, Moltbook (<https://www.astralcodexten.com/p/best-of-moltbook?hide_intro_popup=true>)

When we think about AI improvment we often focus on the role of pretraining, reinforcement learning, inference efficincy improvements.  However, scaffolding or everything else is overlooked. However, it has a incredibly important role in the If we want a full view of what drives progres in language modles we have to start understanding scaffolding. Here we try to startg getting a hold on these effects using data from the Holistic Agent Leaderboard (HAL).  The Holistic Agent leaderboard is a index of model performance on agentic benchmarks which includes accuracy and prices. Importantly, this data runs models on multiple scaffolds (2-3) for each benchmark. Most benchmarks are run using a scaffold that is particular to that benchmark for example on CORE-bench, using the CORE-agent scaffold along with the HAL Genearlist Scaffold. The HAL generalist agents is a agnet based off the smolagent framework where the multiple performs all its actions by writing python code. For instance, the agent searches the intenet by writing python code to search the internet. 

# What is Scaffolding

Scaffolidng is everything that is not included in traditional AI scaling.
This can include:

- Tool affordances available : bash terminal access, file editing tools
- Context Window Management: when is raged used, how is compaction done?
- Memory and summarization
- Prompting:  System prompts, submission prompt instructions, planning prompt
- Reflection and self critique steps
- Token time cost limits, stopping rules
- Agent loop : think → act → observe → repeat


Specialist Scaffolding Has Large Impact on AI Performance
Given the protean nature of scaffolding, it not suprising that its effects vary widely. 
A given scaffold can signficantly improve the performance for one model while hurting the performance for another model.
To look at this effect we graph each models final accuracy as well as the API cost neede to reach this final accuracy. 



- Include Pareto Graphs for a frew benchmarks. 


- Scaffolding effect can be so large that in some cases it makes more sense to benchmark scaffolds rather than models ie it woule be better to have a learderboard of scaffolds rather than a model leaderboard. In the appendix, we include an analysis of variation (ANOVA) where we look at how much performance variation is due to model vs scaffold variation. In some cases, almost as much variation is due scaffolding . To put this in context, it is almost as if the model is an addition on top of the scaffold rather than the scaffold being an adiditon on 


# Scaffolding Effect on Ranking
- Since Scaffolding has interactions model ranking is not useful 
- model Evaluators ideally should test models on multiple scaffolds and take the maximum to get a better ranking 

=======================================

# Claude Scaffold is Suspiciously Strong

# Why This Can Change The Agent Economy

- vertical integration

Claude on Claude Code Harness Performance vs on regular harness
Random Modles on Claude Code Harness vs on regular hanress

4.6 Sonnet

# Conclusion 


# Appendix 

## ANOVA analysis 


## Full Pareto Analysis Graphs 

## Full Vector Analysis Graphs 

## Ranking Correlation Graphs 




==============================================
Junk :
=

System prompt
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
Caching strategy
-


# Does Scaffolding Matter Less At Higher Capabilities

- graphs of capabilitey correlation