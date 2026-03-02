### Task

You are given:

1. the user’s request (QUESTION_CONTENT)
2. the full conversation trajectory (CONVERSATION_HISTORY), including all assistant turns and any tool outputs.

Your job is to evaluate whether the assistant ultimately delivered an **end-to-end usable outcome by the end of the conversation**.
**Completeness is the ONLY evaluation dimension.** Ignore verbosity, writing quality, politeness, and intermediate mistakes.

### Core Principle

At the end of the conversation, **if the user stops right there**, can they achieve their goal **without any essential follow-up**?

* If **YES** → higher completeness score.
* If **NO** → you must name the **completion-critical missing element** (the one thing that prevents success).

### What Counts as “Complete”

The assistant is complete only if it satisfies the user’s goal **end-to-end**, which typically requires:

* **Must-have deliverable** is provided (e.g., final answer, file, code patch, plan, table, steps).
* If actions depend on tools/files, the assistant either:

  * successfully uses them and delivers results, **or**
  * if blocked (tool failure / missing access), provides a **working fallback** (clear manual steps, alternative method, or minimal viable deliverable).
* Includes any essential “last-mile” details: paths, commands, file links, or instructions needed to use the output.

**Do NOT** reward partial attempts unless the outcome is still usable.

### Rating (1–5)

Assign exactly one integer score:

**1 — very incomplete**: No usable outcome; major must-haves missing.
**2 — incomplete**: Some progress, but the user still cannot accomplish the goal.
**3 — partially complete**: Core work attempted; usable only with significant user effort or a key missing piece.
**4 — mostly complete**: Meets most must-haves; only minor omissions or small usability issues remain.
**5 — fully complete**: Fully meets must-haves end-to-end with a usable outcome delivered.

### NEVER Do

* NEVER score tool-call accuracy or penalize “wrong tool usage” unless it **directly prevents completion**.
* NEVER judge style/verbosity/formatting elegance.
* NEVER give credit for intentions (“I will do X later”) unless the deliverable is actually present.
* NEVER assume external actions happened without evidence in the transcript.

## Inputs

### Question Content

```json
{QUESTION_CONTENT}
```

### Conversation History

```json
{CONVERSATION_HISTORY}
```

## Output

Provide your response in the following XML format:

<response>
  <completeness>
    <reasoning>
      <!-- Evaluate if the assistant delivered an end-to-end usable outcome, addressed all requirements, handled tool failures with alternatives, and provided necessary confirmations/paths. -->
    </reasoning>
    <rating><!-- Rating: very incomplete, incomplete, partially complete, mostly complete, fully complete --></rating>
  </completeness>
</response>
