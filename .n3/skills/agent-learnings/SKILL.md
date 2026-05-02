---
name: agent-learnings
description: >
  Use after an agent session that involved multi-turn debugging, iterative
  optimization attempts, ad-hoc script creation, or repeated human corrections
  to agent behavior to extract structured findings for improving existing
  skills or agent configurations. Skip for routine sessions with no struggles
  or discoveries.
user-invocable: false
license: LicenseRef-NvidiaProprietary
metadata:
  author: NVIDIA Corporation
  documentation: https://gitlab-master.nvidia.com/wkong/perf-bot
---

# Agent Learnings

Extract actionable learnings from the current session to improve existing
skills, references, or agent configurations. Output is structured YAML
consumed by agents — not human narratives.

## When to Run

Run when the session included ANY of:

- Agent hit errors and took **3+ turns** to find the real fix
- Agent tried **2+ approaches** to optimize/improve and found what works
- Agent wrote a **reusable script/tool** (20+ lines) that proved effective
- Agent had to **deviate from existing skill guidance** to succeed
- Human **corrected agent behavior 2+ times** in the same problem area

**Produce no output** when the session was routine — all fixes were
single-try, no new techniques discovered, no skill gaps encountered.

## Detection Protocol

Scan the session for five patterns. For each match, produce one finding.

### 1. Debug Resolution

Multi-turn error→fix sequence where the agent struggled to find the root
cause. Extract: exact symptom, root cause, failed approaches (and why each
failed), the actual solution, and which skill should document this.

**Threshold**: 3+ turns from first error to working fix.

### 2. Optimization Discovery

Iterative improve→measure cycle where the agent tried multiple approaches.
Extract: starting state, each approach with its measured result, what worked
and by how much, the underlying insight, and which skill should encode it.

Optimization discoveries often occur WITHIN a broader debugging or analysis
workflow — they don't have to be standalone tasks. Example: after fixing a
profiling issue, the agent analyzes results and iterates on a performance
fix. That iteration is a separate optimization_discovery finding.

**Threshold**: 2+ distinct approaches tried with measurable outcomes.

### 3. Tool/Script Discovery

Agent wrote a script during the session that was executed successfully and
produced useful output. Extract: what it does, I/O format, which skill
directory should house it, and what generalization is needed.

**Threshold**: 20+ lines, successfully executed, reuse potential beyond
this session.

### 4. Skill Gap

Agent followed existing skill guidance but the skill was missing critical
information. The agent had to discover the missing knowledge through trial
and error. Extract: which skill, what was missing, what section to add.

**Threshold**: Agent re-read or referenced the skill AND still needed
additional turns to succeed.

### 5. Repeated Human Correction

Human intervened multiple times to correct the agent's output, approach, or
behavior in the same problem area. The agent completed the task (or a step)
but the human rejected or adjusted the result — indicating the agent's
skills or definition lack knowledge the human keeps supplying manually.

Extract: what the agent did each time, what the human corrected it to, the
pattern across corrections (what knowledge is consistently missing), and
which skill or agent definition should encode that knowledge permanently.

**Threshold**: 2+ human corrections to agent behavior in the same area.

**Exclude**: Cases where the human's initial prompt was vague or incomplete
and the agent made reasonable assumptions. The test: given the instructions
the agent had, *should* it have known better based on its skills and
definition? If yes → finding. If the human simply failed to specify what
they wanted → not a finding.

## Output Format

Output ONLY a YAML code block — no preamble, no analysis, no commentary.
Each detection pattern match produces one finding. Check every pattern
independently; do not skip a pattern because another one already covered
part of the session.

**Valid enum values:**
- `category`: `debug_resolution`, `optimization_discovery`, `script_discovery`, `skill_gap`, `repeated_correction`
- `severity`: `high` (5+ turns, broadly applicable, or 3+ human corrections), `medium` (3-4 turns or 2 corrections), `low` (informational)
- `update_type`: `troubleshooting`, `workflow`, `technique`, `script`, `reference`

Use ONLY these values — do not invent new ones.

```yaml
findings:
  - id: F001
    perfbot_related: true  # Is this finding about PerfBot agents/skills/workflows?
    category: debug_resolution
    severity: high
    summary: "One-line description of the learning"

    # --- Problem & Solution (required for debug_resolution, optimization_discovery, repeated_correction) ---
    problem: |
      What went wrong or what was the agent trying to achieve.
    root_cause: |
      Why this happened — the actual root cause, not symptoms.
    solution: |
      What actually fixed it. Must be a real fix, not a workaround.
    failed_approaches:
      - "Approach: what was tried → why it failed"

    # --- Correction details (required for repeated_correction) ---
    corrections:
      - "Correction 1: agent did X → human corrected to Y (reason)"
      - "Correction 2: agent did A → human corrected to B (reason)"
    human_prompt_unclear: false  # Was the human's initial prompt vague/incomplete?

    # --- Script details (required for script_discovery) ---
    script_purpose: "What the script does"
    script_io: "Input format → Output format"
    generalization_needed: "What changes to make it reusable"

    # --- Target (required for ALL categories) ---
    target_file: "Relative path to an EXISTING skill or config file to update"
    update_type: troubleshooting
    update_section: "Section heading in the target file"
    suggested_update: |
      Exact content ready to insert into the target file.
      Written in the target file's existing style and format.
      Contains the knowledge — not the story of discovering it.

    # --- Evidence (required for ALL categories) ---
    evidence:
      turns_spent: 8
      key_moments:
        - "Turn N: brief description of what happened"
        - "Turn M: breakthrough moment"
```

## Rules

1. **Solutions only, not workarounds.** If an approach didn't fix the root
   cause, it belongs in `failed_approaches`, never in `solution`.

2. **Agent-facing targets only.** Target skills, agent definitions (`.md`
   files), agent configs, CLAUDE.md, or rules files. Never target
   human-facing docs (README, FAQ, user guides).

3. **Match target file style.** `suggested_update` must be formatted to
   match the existing style of `target_file`. Read the file first if unsure.

4. **No narrative.** Write "Use `--trace=osrt` for models >10B params" —
   not "We discovered that after trying several approaches..."

5. **Silence when empty.** If no findings meet detection thresholds, output:
   ```yaml
   findings: []
   ```
   No filler. No "session was clean." No summary of what went well.

6. **One finding per learning.** Don't merge unrelated learnings. Don't
   split one learning across multiple findings.

7. **Evidence is mandatory.** Every finding must link back to specific turns
   in the session. Unsubstantiated findings are noise.

8. **Target must exist.** `target_file` must point to a file that exists —
   a skill SKILL.md, an agent definition `.md` file, CLAUDE.md, or a rules
   file. If no existing file is the natural target, use CLAUDE.md. Never
   fabricate paths to files that don't exist.

9. **Pick the primary category.** When a finding spans categories (e.g.,
   debug_resolution that also reveals a skill_gap), pick the category that
   best represents the actionable update. If two findings would target the
   same file and section, they are the same learning — keep only one.

10. **Verify accuracy.** The agent may have reasoned incorrectly during the
    struggle. Before including technical claims in `suggested_update`,
    verify they are correct — don't propagate mistakes from the session
    into skill updates.

11. **Completeness check.** After generating findings, re-scan the session
    against ALL five detection patterns. A single session commonly produces
    findings across multiple categories (e.g., a debug_resolution AND an
    optimization_discovery AND a repeated_correction). Missing a pattern is
    worse than producing an extra finding.

12. **suggested_update reflects what worked.** The `suggested_update` content
    must be grounded in what actually succeeded in the session. Do not
    include failed approaches as recommended steps in the update — they
    belong only in `failed_approaches`.

13. **Tag PerfBot relevance.** Set `perfbot_related: true` when the finding
    targets PerfBot skills, agents, configuration, or workflows. Set `false`
    when the finding is about unrelated work done in the same session (e.g.,
    general coding, other projects). Only PerfBot-related findings are
    uploaded to the central server.
