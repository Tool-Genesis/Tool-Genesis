from __future__ import annotations
from typing import Optional, Dict


class TurnTracker:
    """
    Lightweight turn controller for agent loops.

    Features:
      - next_turn(): advance the loop and enforce max_turns
      - reset(): reset counters (optionally update max_turns)
      - remaining(): how many turns are left
      - meta_tip(): emit an English "[meta-info]" guidance message tailored to remaining turns
      - stop(): terminate early with a reason
      - done(): whether the loop is finished
      - summary(): small state snapshot
      - set_return_control(): control whether the tracker should return/continue

    Usage:
        tracker = TurnTracker(max_turns=3, name="world-model-pipeline")
        while tracker.next_turn():
            print(tracker.meta_tip(action_hint="Run eval, then fix failing cases."))
            # ... your agent orchestration ...
            # if success:
            #     tracker.stop(reason="success")
            #     break
            # if some_condition:
            #     tracker.set_return_control(enable_return=False)  # Disable return
    """

    def __init__(self, max_turns: int, *, name: str = "", enable_return: bool = True) -> None:
        assert max_turns >= 1, "max_turns must be >= 1"
        self.name = name
        self.max_turns = int(max_turns)
        self.current_turn = 0   # 0 before the first call to next_turn()
        self.terminated = False
        self.reason: Optional[str] = None
        self.enable_return = enable_return  # Control whether return is allowed

    # ---------- control ----------
    def reset(self, max_turns: Optional[int] = None, enable_return: Optional[bool] = None) -> "TurnTracker":
        """Reset counters; optionally update max_turns and enable_return."""
        if max_turns is not None:
            assert max_turns >= 1, "max_turns must be >= 1"
            self.max_turns = int(max_turns)
        if enable_return is not None:
            self.enable_return = enable_return
        self.current_turn = 0
        self.terminated = False
        self.reason = None
        return self

    def next_turn(self) -> bool:
        """
        Advance to the next turn if allowed.
        Returns True if the loop should continue, False otherwise.
        
        If enable_return is False, always returns False (forces early termination).
        """
        if not self.enable_return:
            self._terminate("return_disabled")
            return False
            
        if self.terminated:
            return False
        if self.current_turn >= self.max_turns:
            self._terminate("max_turns")
            return False
        self.current_turn += 1
        return True

    def stop(self, *, reason: str = "success") -> None:
        """Terminate early (e.g., success/abort/error)."""
        self._terminate(reason)

    def done(self) -> bool:
        """True if terminated, return is disabled, or the limit has been reached."""
        return not self.enable_return or self.terminated or self.current_turn >= self.max_turns

    def set_return_control(self, enable_return: bool) -> None:
        """Control whether the tracker should allow returns/continuation."""
        self.enable_return = enable_return
        if not enable_return and not self.terminated:
            self._terminate("return_disabled")

    # ---------- queries ----------
    def remaining(self) -> int:
        """Number of remaining turns (>= 0). Returns 0 if return is disabled."""
        if not self.enable_return:
            return 0
        return max(self.max_turns - self.current_turn, 0)

    def meta_tip(self, *, action_hint: Optional[str] = None) -> str:
        """
        Emit an English "[meta-info]" guidance message based on remaining turns.

        Rules:
          - Return disabled -> indicate that execution is disabled
          - 0 left  -> urge to produce the final answer now.
          - 1 left  -> urge to move directly to the final answer now.
          - >=2 left-> encourage progress and show how many turns remain.

        Optionally append a short "Next step" hint.
        """
        if not self.enable_return:
            base = ""
        else:
            rem = self.remaining()
            if rem <= 0:
                base = "[meta-info] 0 turns left. Produce the final answer now with <final> ##you final answer </final>, and don't using tools, strictly following the requirements."
            elif rem == 1:
                base = "[meta-info] 1 turn left. Move directly to the final answer now as required."
            else:
                base = f"[meta-info] {rem} turns remaining. Keep solving and make concrete progress this turn.If you want to stop, please only use <final> ##you final answer </final> and Don't use <analysis>."
        
        if action_hint and self.enable_return:
            base += f" Next step: {action_hint}"
        return base

    def summary(self) -> Dict[str, object]:
        """Compact state snapshot for logging/telemetry."""
        return {
            "name": self.name,
            "max_turns": self.max_turns,
            "current_turn": self.current_turn,
            "remaining": self.remaining(),
            "terminated": self.terminated,
            "reason": self.reason,
            "enable_return": self.enable_return,
        }

    # ---------- internals ----------
    def _terminate(self, reason: str) -> None:
        self.terminated = True
        self.reason = reason