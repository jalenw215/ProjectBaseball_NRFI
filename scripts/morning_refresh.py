from __future__ import annotations

from nrfi_predictor.runner import PipelineRunner


def main() -> int:
    runner = PipelineRunner(progress=print)
    results = runner.morning_refresh()
    for result in results:
        print(f"{result.status.upper()} {result.name}: {result.detail}")
    return 1 if any(result.status != "ok" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
