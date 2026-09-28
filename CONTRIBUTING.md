# Contributing

Keep changes scoped to one scientific or engineering concern. New objectives need a focused loss test, an explicit supervision contract, checkpoint metadata, and a student-only evaluation path. Do not change candidate construction in an objective comparison without changing the experiment name and recording new fingerprints.

Before opening a change:

```bash
python -m ruff check src tests
python -m pytest
python -m compileall -q src
```
