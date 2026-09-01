# Calibration v11 gap analysis

The v11 recipe is based on two independent checkpoints quantized with the same v10 calibration
corpus and evaluated on the same 88-case agent/tool reliability suite.  Only aggregate behavior
labels were used; evaluation prompts and answers are not included in calibration data.

| Result | Run A | Run B |
|---|---:|---:|
| Pass | 68 | 72 |
| Partial | 17 | 16 |
| Fail | 3 | 0 |
| Points | 153/176 (86.9%) | 160/176 (90.9%) |

Twelve cases were weak in both runs.  Their shared failure families are:

- completing all phases of long research and planning workflows;
- acting after a condition or newly revealed information becomes sufficient;
- polling asynchronous work until a terminal result and surfacing that result;
- performing a required search instead of answering from prior knowledge;
- retaining user corrections and accumulated constraints through the final action;
- exact schema semantics and restraint from unnecessary tools or extra output;
- discovery and ambiguous-commit verification before exactly-once provisioning;
- read-before-write without redundant follow-up calls.

Run-specific severe failures add lower-weight coverage for capability boundaries, safe multi-turn
mutation, untrusted-output summarization without echoing payloads, and fresh relationship
verification when memory may be stale.

## v11 changes

`calib_sources --profile v11` inherits the v10 native tool trajectories and adds a separate
`workflow_reliability` category with 16 synthetic scenario families and 2,048 deterministic
variants.  The examples use unrelated entities and values so they teach the control-flow pattern
without reproducing benchmark items.

The 1,048,576-token main mix allocates 20% to this category while retaining code, multilingual,
general, schema/function, math, injection, and public ToolACE coverage.  The 64 × 8192-token IGS
corpus uses 75% general long context and 25% workflow trajectories.  MaCa lengths, candidate
reserve, COLA/ACDM selection, MR-GPTQ settings, and histogram-based IGS merging remain unchanged
from v10 so the calibration revision stays attributable.
