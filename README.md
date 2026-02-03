# Wheelchair Rugby Lineup Modeling

This repo contains the full workflow: exploratory analysis, model training/tuning, optimization, and a simple decision-support dashboard.

## Setup

1) Create and activate a virtual environment  
2) Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

Required files (already in `Data/`):
- `Data/player_data.csv`
- `Data/stint_data.csv`

Most scripts will auto-fall back to `Data/` if root-level CSVs are not found.

## EDA

Open the notebook:
- `EDA/Wheelchair_Rugby_EDA.ipynb`

## Model Tuning (per model)

Each model can tune its own hyperparameters using **5-fold CV on the training split**, then report **held-out test** performance. Best values are written to **two places**:
1) Central config: `MLModel/model_config.yaml` (sectioned)
2) Per-model config: same filename as the model (e.g., `MLModel/Models/RidgeRegression.yaml`)

Run tuning using module paths (recommended):

```bash
python -m MLModel.Models.baseline_regression --tune
python -m MLModel.Models.baseline_regression_plus_home --tune
python -m MLModel.Models.RidgeRegression --tune
python -m MLModel.Models.ElasticNetRegression --tune
python -m MLModel.Models.RandomForestModel --tune
```

You can override tuning grids using CLI flags (see `--help` on each script).  
Example with explicit CV folds:

```bash
python -m MLModel.Models.RidgeRegression --tune --cv_folds 5
```

## Benchmark All Models (tune + evaluate)

Run the full sweep (tune all with CV, then benchmark all on held-out test split):

```bash
python MLModel/tune_and_benchmark.py
```

To skip tuning and just benchmark with saved configs:

```bash
python MLModel/tune_and_benchmark.py --skip_tune --markdown
```

Optional CSV output:

```bash
python MLModel/tune_and_benchmark.py --out MLModel/benchmarks_all.csv
```

Default benchmark output path (even without `--out`):
- `MLModel/benchmarks_all.csv`

## Validate a Single Model

Use the validation CLI to evaluate a specific model type:

```bash
python Optimizer/validate_model.py --model ridge --config_section ridge
python Optimizer/validate_model.py --model elasticnet --config_section elasticnet
python Optimizer/validate_model.py --model random_forest --config_section random_forest
```

## Ridge No-Synergy Effects Report

Generate a report-style summary of the **Ridge (no synergies)** model effects (player + home/away + opponent):

```bash
python MLModel/RidgeNoSynergyEffects/ridge_no_synergy_effects_report.py
```

What it does:
- Fits `MLModel/Models/RidgeRegressionContext.py` using saved best params by default (or an overridden `--alpha`).
- Prints model metrics and the top-ranked effects by:
  - `mean_abs_contrib` (practical contribution strength), and
  - `abs_coef` (per-activation effect size).
- Prints general team-context summaries:
  - league-wide home effect (`home_adv_global`),
  - top teams by `home_adv_coef`,
  - strongest opponents via `opponent_strength_index = -opp_coef`,
  - combined context index (`home_adv_coef - opp_coef`).

Default outputs:
- `MLModel/RidgeNoSynergyEffects/ridge_no_synergy_effect_strengths.csv` (all feature strengths and rankings)
- `MLModel/RidgeNoSynergyEffects/ridge_no_synergy_group_strengths.csv` (group totals: player/home/opponent)
- `MLModel/RidgeNoSynergyEffects/ridge_no_synergy_general_effects_by_team.csv` (team-level home/opponent summaries)

Optional output:
- `MLModel/RidgeNoSynergyEffects/ridge_no_synergy_example_stint_contributions.csv` when `--example_stint_idx` is provided

Useful flags:
- `--top_n 15`
- `--alpha 20`
- `--example_stint_idx 0`

## Lineup Optimizer (CLI)

Score lineups and produce a handoff table:

```bash
python Optimizer/lineup_optimizer_canada.py \
  --context home \
  --opponent Japan \
  --handoff_lineup Canada_p1 Canada_p3 Canada_p7 Canada_p10
```

## Counterfactual Demo (All Canada Games)

Run a counterfactual baseline-vs-ML replay across every original Canada game:

```bash
python Optimizer/CaseStudy_OriginalGames/canada_counterfactual_all_games.py --B 300
```

What it does:
- Uses the tuned Ridge no-synergy model (`MLModel/Models/RidgeRegressionContext.py`).
- Replaces each historical Canada lineup with the model-chosen best eligible lineup.
- Computes point-estimate counterfactual goal differential per game.
- Bootstraps model refits to produce 95% CI and win probability for each game.

Location:
- All replay/counterfactual scripts, CSVs, and images are stored in `Optimizer/CaseStudy_OriginalGames/`.

Defaults:
- Lineups are restricted to players who actually appeared in that game.
- Use `--use_full_roster` to allow all Canada players in every game.

Reproducibility (from repo root):
```bash
# 1) Recompute counterfactual results across all original Canada games
python Optimizer/CaseStudy_OriginalGames/canada_counterfactual_all_games.py --B 300 --seed 42

# 2) Recreate the win-flip figure from the generated CSV
python Optimizer/CaseStudy_OriginalGames/plot_counterfactual_win_flip.py
```

Outputs:
- `Optimizer/CaseStudy_OriginalGames/canada_counterfactual_all_games.csv`
- `Optimizer/CaseStudy_OriginalGames/canada_counterfactual_lost_games.csv`
- `Optimizer/CaseStudy_OriginalGames/counterfactual_win_flip_plot.png`

## Dashboard

Launch the Streamlit app:

```bash
streamlit run DecisionSupportSystem/coach_dashboard.py
```

## Config Files

Central config:
- `MLModel/model_config.yaml` (sectioned per model)

Per-model configs created by tuning:
- `MLModel/Models/baseline_regression.yaml`
- `MLModel/Models/baseline_regression_plus_home.yaml`
- `MLModel/Models/RidgeRegression.yaml`
- `MLModel/Models/ElasticNetRegression.yaml`
- `MLModel/Models/RandomForestModel.yaml`
