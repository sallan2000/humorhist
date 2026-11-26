#!/usr/bin/env python3
"""Disabled: the legacy bulk re-screen could erase scores and duplicate requests."""


def main() -> int:
    print("Re-screening is disabled: existing scores have not been changed. "
          "Use humorhist screen-status to inspect blocked attempts; see README "
          "for safe recovery. A safe re-screen workflow is not yet available.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
