# Working Title

Inference-Time Latent Memory Management for Long-Horizon LLM Agents

# Core Claim

We propose a session-level latent memory bank for MemGen-style LLM agents. The memory bank stores, retrieves, updates, and reuses latent memories during inference, enabling the model to better use historical information long-context reasoning tasks 

# Motivation

Large language model agents often need to use information from previous turns or earlier parts of a long context. Existing latent-memory methods such as MemGen can generate latent memories during reasoning, but these latent memories are usually consumed locally and are not explicitly managed across the session. This limits their ability to support reusable memory over multiple reasoning steps.

# Key Idea

We extend MemGen with a session-local latent memory bank. When the Weaver generates latent memory, the memory bank stores it. At later reasoning steps, the model can retrieve relevant latent memories from the bank and inject them into the Reasoner. The memory bank also supports update, replacement, and reset operations.

# Contributions

1. We introduce an inference-time latent memory management mechanism for MemGen-style LLM agents.

2. We design a session-local latent memory bank with explicit write, retrieval, update, replacement, and reset operations.

3. We evaluate the proposed mechanism on long-context reasoning tasks, analyzing both task performance and internal memory behavior.

# Research Questions

RQ1. Does the proposed memory bank preserve the original MemGen behavior when disabled?

RQ2. Does the memory bank produce meaningful write, retrieval, and update behavior during inference?

RQ3. Does session-level latent memory reuse improve performance on long-context reasoning tasks?

RQ4. Which memory-bank design choices, such as threshold, top-k, capacity, and replacement policy, matter most?

# Paper Structure

1. Introduction
2. Related Work
3. Background: MemGen
4. Method
5. Experiments
6. Analysis
7. Limitations
8. Conclusion