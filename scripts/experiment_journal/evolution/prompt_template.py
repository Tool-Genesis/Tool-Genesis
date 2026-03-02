"""
Evolution Prompt Templates for Experiment B.

Provides prompt templates for each round of tool evolution,
incorporating task-level feedback into the generation prompt.
"""

# ---------------------------------------------------------------------------
# Round 0: Initial generation (reuse existing pipeline prompt)
# ---------------------------------------------------------------------------

ROUND_0_SYSTEM = """\
You are a developer building MCP (Model Context Protocol) tool servers.
Given a scenario description, you must produce a complete, runnable Python
MCP server that implements all the required tools.

Output only the Python source code for the server, enclosed in a single
```python ... ``` block.
"""

ROUND_0_USER = """\
Build an MCP server for the following scenario:

{requirement}
"""

# ---------------------------------------------------------------------------
# Evolution rounds (round >= 1): feedback-driven improvement
# ---------------------------------------------------------------------------

EVOLUTION_SYSTEM = """\
You are a developer iterating on an MCP (Model Context Protocol) tool server.
You previously built a server, and it has been tested with real tasks.
Based on the test feedback, you must produce an improved version.

You may:
- Fix bugs in existing tools
- Modify tool parameters or return formats
- Add new tools to handle missing capabilities
- Remove unnecessary or broken tools
- Restructure the implementation for better reliability

Output only the complete updated Python source code, enclosed in a single
```python ... ``` block. The output must be a fully self-contained, runnable
MCP server — do not output diffs or partial code.
"""

EVOLUTION_USER = """\
You previously built an MCP server to handle the following scenario:

{requirement}

Your current implementation (version {round}):

```python
{current_server_code}
```

After testing with real tasks, here is the feedback:

{feedback_text}

Please analyze the failures and produce an improved version of the MCP server.
Focus on fixing the issues identified in the feedback while keeping the
working parts intact. Output the complete updated server code.
"""

# ---------------------------------------------------------------------------
# Feedback-only variant (no code shown, let model regenerate from scratch)
# ---------------------------------------------------------------------------

EVOLUTION_USER_NO_CODE = """\
You previously built an MCP server to handle the following scenario:

{requirement}

After testing version {round} with real tasks, here is the feedback:

{feedback_text}

Based on this feedback, build an improved MCP server from scratch.
Ensure all identified issues are addressed. Output the complete server code.
"""


def build_round0_messages(requirement: str) -> list:
    """Build prompt messages for initial generation (round 0)."""
    return [
        {"role": "system", "content": ROUND_0_SYSTEM},
        {"role": "user", "content": ROUND_0_USER.format(requirement=requirement)},
    ]


def build_evolution_messages(
    requirement: str,
    current_code: str,
    feedback_text: str,
    round_num: int,
    include_code: bool = True,
) -> list:
    """
    Build prompt messages for evolution round.

    Args:
        requirement: Original scenario description.
        current_code: Current server implementation.
        feedback_text: Formatted feedback from feedback_collector.
        round_num: Current round number (>= 1).
        include_code: Whether to include the current code in the prompt.
    """
    if include_code:
        user_content = EVOLUTION_USER.format(
            requirement=requirement,
            round=round_num,
            current_server_code=current_code,
            feedback_text=feedback_text,
        )
    else:
        user_content = EVOLUTION_USER_NO_CODE.format(
            requirement=requirement,
            round=round_num,
            feedback_text=feedback_text,
        )

    return [
        {"role": "system", "content": EVOLUTION_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def build_evolution_history_messages(
    requirement: str,
    code_versions: list,
    feedback_history: list,
    round_num: int,
) -> list:
    """
    Build prompt with full evolution history (all rounds).

    This variant includes the complete conversation history:
    round 0 code, round 0 feedback, round 1 code, round 1 feedback, ...

    Args:
        requirement: Original scenario description.
        code_versions: List of code strings [v0, v1, ..., v_{round-1}].
        feedback_history: List of feedback text strings [fb_0, fb_1, ...].
        round_num: Current round number.
    """
    messages = [{"role": "system", "content": EVOLUTION_SYSTEM}]

    history_parts = []
    history_parts.append(f"## Original Scenario\n\n{requirement}")

    for i, (code, fb) in enumerate(zip(code_versions, feedback_history)):
        history_parts.append(f"\n## Version {i} Implementation\n\n```python\n{code}\n```")
        history_parts.append(f"\n## Feedback after Version {i}\n\n{fb}")

    history_parts.append(
        f"\n## Task\n\nBased on the feedback history above, produce version {round_num} "
        f"of the MCP server. Address all identified issues, especially those from the "
        f"most recent round. Output the complete updated server code."
    )

    messages.append({"role": "user", "content": "\n".join(history_parts)})
    return messages
