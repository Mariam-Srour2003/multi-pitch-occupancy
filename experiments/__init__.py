"""Thesis experiments. A package so that one experiment can import another.

Three scripts here import from a sibling - `false_play_significance` from
`h3_with_false_play`, `rescore_false_play` from `preprocess_search`, `benchmark_v2` from
`prompt_search` - and until this file existed **none of them could run**. Executing
`python experiments/x.py` puts `experiments/` on `sys.path`, not the repository root, so
`import experiments.h3_with_false_play` raised `ModuleNotFoundError`. The tests never saw
it because `pyproject.toml` sets `pythonpath = ["."]` for pytest alone.

`reproduce_all.py` therefore invokes every stage as `python -m experiments.<name>`, which
puts the repository root on the path and makes the two forms of "run this experiment" the
same thing. `tests/test_reproduce_all.py` asserts every experiment module imports under
exactly that invocation, because two of them had been unreachable for weeks while the
reproduction pipeline reported both stages as done - their CSVs existed from an earlier
run, and nothing checked that the command which produced them still worked.

Nothing in `src/pitch_occupancy/` imports from here. The dependency points one way.
"""
