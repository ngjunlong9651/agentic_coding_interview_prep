# Agentic Coding Interview Prep

Preparation for a round in the format "2 hours of coding with AI tools, then a 1 hour interview about what you built".
It has three tracks: build an agent, drive an AI coding tool well, and prove and defend the result.

## Contents
| File | What it is |
|---|---|
| `agentic_coding_interview_prep.ipynb` | The lecture notebook. Part 1 builds an agent from scratch (model call, messages, tools, the loop) with a toy shop; Parts 2 to 4 scale it to a coding agent, make it safe and measure it; Parts 5 to 9 cover the 2 hour game plan, production questions, interview defence and drills |
| `agent.py` | A minimal coding agent on the raw Claude Messages API: sandboxed tools, the loop with budgets and approval, a trace, and a `ScriptedClient` for offline runs |
| `practice_repo/` | A tiny shopping-cart module with planted bugs and a pytest suite. Every lab copies it into `runs/` first, so the original stays buggy |
| `requirements.txt` | anthropic, pytest, ipykernel, pandas, matplotlib |

Parts 1 and 2 call the real Claude API (`claude-opus-5-5`), so set `ANTHROPIC_API_KEY` first; together they cost well under USD 1.
A fake client (`ScriptedClient`) is used only where a failure must happen on cue (Part 3) and for tests (Part 4).

## Slides
A companion slide deck lives on the author's Claude account (visible only if the author has shared it): https://claude.ai/artifact/TakAVAzfmzNb9zqB8hrEgS

## Setup
```bash
uv venv .venv
uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name agentic-interview --display-name "Python (agentic-interview)"
export ANTHROPIC_API_KEY=...   # optional, only for LIVE cells
```
Open the notebook and select the **Python (agentic-interview)** kernel.

## Topics
The 2 hour timeline and prompting loop, CLAUDE.md template, the agent loop and its four rules, manual loop vs tool runner vs frameworks,
tool design, errors as data, budgets, human approval, prompt injection, unit and scripted-loop tests, eval harness with validated graders,
pass rate and cost metrics, LLM as judge, cost and latency levers, security, observability, the 5 minute walkthrough, a question bank,
and four mock drill briefs with a scoring rubric.

## Series
Part of a small AI engineering learning series: Hybrid RAG, Harness Engineering, LLM Evals, Tool Use and MCP, Agentic Coding Interview Prep.
