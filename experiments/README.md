# Experiments

One script per thesis experiment, named for the task it implements (`wp4_t1_split_comparison.py`).
Each script must:

- take its split by **name** from `results/splits/`, never re-randomise inline;
- set every seed it uses, and record it in the output;
- write results to a CSV in `results/`, appending rather than overwriting;
- add one line to `results/EXPERIMENT_LOG.md` — date, task id, command, result file, finding;
- be re-runnable from the feature cache without recomputing embeddings.

Nothing here is imported by `src/pitch_occupancy/`. The dependency points one way: experiments
use the library, the library never uses experiments. Anything an experiment needs twice belongs
in the package instead.

`tools/reproduce_all.py` (WP0-T11) runs the whole set from the manifest and the cached features,
and must regenerate every table and figure in the thesis.
