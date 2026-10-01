# AGENTS_Figure — Scientific Visualization and Publication Figure Standard

> **Scope**: General scientific visualization standard for all projects.  
> **Primary use**: Machine learning, ADMET, PBPK, cheminformatics, pharmacokinetics, model comparison, ablation studies, statistical analysis, and manuscript figures.  
> **Language**: All figure-visible text must be in English unless a specific deliverable explicitly requires another language.  
> **Principle**: Scientific correctness > readability > aesthetics.

---

# 1. Purpose

This document defines the project-wide rules for generating:

- exploratory figures;
- model evaluation figures;
- statistical figures;
- ablation figures;
- data-distribution figures;
- publication-ready manuscript figures;
- supplementary figures;
- presentation-ready scientific figures.

All agents generating scientific figures must follow this standard unless a task-specific document explicitly overrides it.

The goals are:

1. scientific accuracy;
2. reproducibility;
3. visual consistency;
4. accessibility;
5. publication-quality layout;
6. no fabricated or unsupported data;
7. clear separation between measured, predicted, derived, and illustrative information.

---

# 2. Non-negotiable scientific integrity rules

## 2.1 Never fabricate data

Agents must never:

- invent missing values;
- generate plausible-looking experimental results;
- create fake replicates;
- manufacture uncertainty estimates;
- manually alter model metrics;
- visually add points not present in source data;
- modify observations to improve appearance;
- infer unavailable measurements and present them as measured;
- replace missing results with assumed values;
- draw unsupported significance annotations.

If required data do not exist:

```text
DO NOT PLOT AS REAL DATA.
```

Instead:

- omit the unsupported element;
- mark it as unavailable;
- report that the required data are missing;
- generate the rest of the figure if scientifically valid.

---

# 3. Data provenance

Every scientific figure must be traceable to its source data.

Agents should be able to identify:

```text
figure
  ↓
plotting script
  ↓
processed plotting table
  ↓
source result/data files
  ↓
original experiment/model run
```

For publication-quality figures, create a figure manifest whenever practical.

Recommended:

```text
figures/
├── scripts/
├── data/
├── output/
└── manifests/
```

Example:

```text
figures/
├── scripts/
│   └── fig_baseline_mae.py
├── data/
│   └── fig_baseline_mae.csv
├── output/
│   ├── fig_baseline_mae.pdf
│   ├── fig_baseline_mae.svg
│   └── fig_baseline_mae.png
└── manifests/
    └── fig_baseline_mae.json
```

---

# 4. Figure manifest

Recommended fields:

```json
{
  "figure_id": "Fig_01",
  "title": "Baseline model comparison",
  "script": "figures/scripts/fig_baseline_mae.py",
  "source_files": [],
  "source_hashes": {},
  "generated_at": null,
  "git_commit": null,
  "python_version": null,
  "library_versions": {},
  "filters": [],
  "transformations": [],
  "statistics": [],
  "output_files": []
}
```

Any filtering or transformation must be explicitly recorded.

---

# 5. Source-of-truth rule

Plots must be generated from machine-readable source files whenever possible.

Preferred:

```text
CSV
TSV
JSON
Parquet
NPY
NPZ
database query output
model prediction files
```

Avoid manually copying values into Python source code.

Bad:

```python
mae = [0.31, 0.28, 0.25]
```

when these numbers already exist in result files.

Preferred:

```python
df = pd.read_csv(result_path)
```

Hard-coded values are acceptable only for:

- fixed reference thresholds;
- theoretical constants;
- documented literature values;
- explicitly labeled schematic/example figures.

Their source must be documented.

---

# 6. Measured vs predicted vs derived data

Agents must clearly distinguish:

```text
Measured
Predicted
Derived
Reference
Literature
Illustrative
```

Do not use visual styling that implies these are equivalent.

For example:

- measured values: solid markers;
- predictions: open markers or separate visual encoding;
- reference line: neutral dashed line;
- derived estimates: clearly documented in legend/caption.

---

# 7. No hidden data manipulation

Agents must not silently apply:

- clipping;
- winsorization;
- smoothing;
- interpolation;
- outlier removal;
- normalization;
- logarithmic transformation;
- rescaling;
- averaging;
- aggregation;
- sample exclusion.

If scientifically required, the operation must be:

1. performed explicitly;
2. documented;
3. reflected in the figure manifest;
4. stated in the caption where relevant.

---

# 8. Axis integrity

Never manipulate axes to visually exaggerate effects.

Agents must:

- use scientifically appropriate axis limits;
- clearly label log-transformed axes;
- explicitly indicate broken axes;
- avoid unnecessary axis truncation;
- avoid misleading aspect ratios;
- avoid reversing axes unless scientifically conventional.

For logarithmic axes:

```text
log scale must be explicitly visible or stated.
```

For model targets already stored on log scale, do not label them as raw physical quantities.

---

# 9. Default project color system

Use one consistent palette across projects.

Default palette is based on a color-vision-friendly scientific palette.

```text
Primary blue        #0072B2
Sky blue            #56B4E9
Bluish green        #009E73
Orange              #E69F00
Vermillion          #D55E00
Reddish purple      #CC79A7
Yellow              #F0E442
Black               #000000
Neutral gray        #7A7A7A
Light gray          #D9D9D9
```

---

# 10. Semantic color rules

Whenever possible, colors should have stable meanings across projects.

Recommended:

```text
Primary method/model       #0072B2
Secondary comparison       #D55E00
Alternative method         #009E73
Additional comparison      #CC79A7
Reference/baseline         #7A7A7A
Null/reference line        #000000
```

Do not randomly change a model's color between figures.

Example:

If Morgan is blue in Figure 2, Morgan should remain blue in Figures 3–6 unless there is a strong reason otherwise.

---

# 11. Color accessibility

Never rely on color alone.

Different groups should also be distinguishable using one or more of:

- marker shape;
- line style;
- hatch;
- fill/open markers;
- direct labels.

Avoid red-vs-green as the sole comparison.

All figures should remain interpretable:

```text
in grayscale
and
for common color-vision deficiencies.
```

---

# 12. Yellow usage

The default yellow:

```text
#F0E442
```

must not normally be used for:

- thin lines;
- small text;
- small scatter markers on white backgrounds.

It may be used for:

- large filled regions;
- highlighted areas;
- heatmap categories with sufficient contrast.

---

# 13. Sequential and diverging color maps

For continuous data, prefer perceptually uniform colormaps.

Recommended sequential:

```text
viridis
cividis
magma
plasma
```

Preferred default:

```text
viridis
```

For values with a meaningful center, use a diverging map.

Example:

```text
negative ← 0 → positive
```

The center must correspond to the scientifically meaningful reference value.

Avoid:

```text
jet
rainbow
```

unless required by an established domain convention.

---

# 14. Typography

Default figure font:

```text
Arial
```

Fallback:

```text
Helvetica
DejaVu Sans
Liberation Sans
```

Use the same font family throughout one figure and preferably throughout the manuscript.

---

# 15. Text language

All visible plot annotations should normally be English:

```text
axis labels
titles
legend
panel labels
annotations
colorbar labels
category names
statistical labels
```

Source code comments may use English or the project's working language.

---

# 16. Font hierarchy

Recommended starting point for publication figures:

```text
Panel label        9–11 pt, bold
Axis label         8–10 pt
Tick label         7–9 pt
Legend             7–9 pt
Annotation         7–9 pt
```

Exact sizes may be adjusted according to final physical figure dimensions.

Never judge font size only from a large computer-screen preview.

The final figure must remain readable after reduction to publication size.

---

# 17. Figure dimensions

Use physical dimensions rather than arbitrary large pixel canvases.

Recommended starting sizes:

```text
Single-column figure: ~85–90 mm wide
Double-column figure: ~170–180 mm wide
```

Exact dimensions must follow the target journal if known.

For Python:

```python
MM_TO_INCH = 1 / 25.4
```

Example:

```python
width = 90 * MM_TO_INCH
```

---

# 18. Output formats

For line plots, scatter plots, bar plots, diagrams, and other vector-compatible graphics:

Always prefer:

```text
PDF
SVG
```

Also generate a preview:

```text
PNG
```

Recommended:

```text
PNG >= 300 dpi
```

For line-heavy publication graphics:

```text
600 dpi PNG
```

may be generated as an additional raster export.

Do not rasterize text and vector elements unnecessarily.

---

# 19. Standard output set

Publication figures should normally produce:

```text
figure_name.pdf
figure_name.svg
figure_name.png
```

Optional:

```text
figure_name.tiff
```

when required by the target journal.

---

# 20. Figure background

Default:

```text
white
```

Avoid:

- gradients;
- decorative shadows;
- dark backgrounds;
- unnecessary colored panels.

Scientific figures should prioritize information density and clarity.

---

# 21. Grid lines

Use grid lines sparingly.

Preferred:

```text
light horizontal grid only
```

when they aid quantitative comparison.

Avoid strong grids that compete with the data.

---

# 22. Spines and borders

Prefer minimal framing.

Typical publication style:

```text
left spine     visible
bottom spine   visible
top spine      removed
right spine    removed
```

unless a complete box is scientifically useful.

---

# 23. Legends

Legends must not cover:

- data points;
- regression lines;
- error bars;
- confidence intervals;
- annotations.

Preferred strategies:

1. place legend in unused plot space;
2. place legend outside axes;
3. use direct labeling where possible;
4. increase figure width if necessary.

Agents must inspect the rendered figure.

Do not assume automatic legend placement is safe.

---

# 24. Label collision

No visible overlap is allowed among:

```text
tick labels
axis labels
panel labels
legend
annotations
data labels
colorbars
titles
```

Agents should explicitly check the rendered output.

Acceptable solutions:

- increase figure size;
- adjust margins;
- rotate tick labels;
- wrap labels;
- move legend;
- reduce annotation density;
- use direct labels selectively;
- use constrained/tight layout carefully.

Do not solve overlap by making all fonts unreadably small.

---

# 25. Titles

Main manuscript plots often do not require large titles inside the figure.

Prefer:

```text
information in figure caption
```

rather than a large chart title.

For exploratory figures, titles are acceptable.

---

# 26. Multi-panel figures

Use lowercase panel labels:

```text
a
b
c
d
```

Place them consistently near the upper-left corner of each panel.

Do not use inconsistent forms such as:

```text
A
(B)
Panel C
```

within the same manuscript.

---

# 27. Panel alignment

Multi-panel figures should have:

- aligned axes where appropriate;
- consistent panel widths;
- consistent font sizes;
- balanced whitespace;
- consistent legend style;
- consistent color mappings.

Panels should form one scientific narrative rather than being combined only to save space.

---

# 28. Chart selection principles

Choose the plot according to the scientific question.

## Distribution

Prefer:

```text
histogram
ECDF
violin + raw points
boxplot + raw points
```

## Category comparison

Prefer:

```text
dot plot
point-range plot
box/violin plot
bar plot only when appropriate
```

## Relationship

Prefer:

```text
scatter plot
hexbin/density scatter for large N
```

## Time/order

Prefer:

```text
line plot
```

## Matrix

Prefer:

```text
heatmap
```

---

# 29. Avoid unnecessary pie charts

Pie charts should normally not be used for precise scientific comparisons.

Prefer:

```text
bar chart
dot plot
stacked bar
```

when quantitative comparison matters.

---

# 30. Avoid unnecessary 3D plots

Do not use 3D bars, 3D pie charts, or perspective effects for ordinary quantitative data.

They distort perceived magnitude and add visual clutter.

3D visualization should only be used when the third spatial dimension represents actual scientific information.

---

# 31. Raw data visibility

When sample size permits, show individual observations.

For example:

```text
boxplot + jittered raw points
```

is usually more informative than a boxplot alone.

Do not hide sample distribution behind summary statistics when the raw observations can reasonably be displayed.

---

# 32. Error bars

Every error bar must have a defined meaning.

Examples:

```text
mean ± SD
mean ± SEM
95% CI
min–max
interquartile range
variation across model seeds
variation across data splits
```

Never draw generic error bars without specifying their definition.

---

# 33. Independent runs must be distinguished

For machine-learning results, distinguish:

```text
split variability
model-seed variability
bootstrap uncertainty
cross-validation variability
experimental replicate variability
```

Do not combine these into one generic `"error"`.

---

# 34. Sample size

Where scientifically relevant, provide:

```text
n = ...
```

The meaning of `n` must be clear.

Examples:

```text
n = compounds
n = independent experiments
n = model seeds
n = splits
```

---

# 35. Statistical significance

Do not automatically add:

```text
*
**
***
****
```

Agents may only add significance annotations when:

1. an appropriate statistical test has actually been performed;
2. the tested groups are clearly defined;
3. multiple-testing correction is handled where appropriate;
4. exact or thresholded p-values are documented.

Prefer exact p-values where space permits.

---

# 36. Correlation plots

When showing predicted vs observed values:

Recommended elements:

```text
scatter points
identity line y = x
axis labels with target definition
n
primary metrics
```

Optional:

```text
regression line
confidence interval
```

Do not replace the identity line with a fitted regression line.

They answer different questions.

---

# 37. Predicted-vs-observed axis consistency

For parity plots:

```text
x = Observed
y = Predicted
```

should be the default throughout the project.

Use the same x/y limits when practical.

Use equal aspect ratio when the scientific interpretation benefits from it.

---

# 38. Residual plots

Recommended definition:

```text
Residual = Predicted - Observed
```

unless the project explicitly defines the opposite.

Record the definition.

Residual plots should include:

```text
horizontal reference at 0
```

---

# 39. Model comparison figures

For model-performance comparison, prefer showing:

```text
individual run values
+
mean
+
uncertainty
```

instead of only bars.

For example:

```text
5 split MAE values
+
mean ± SD
```

This exposes run-to-run variation.

---

# 40. Metric direction

Clearly distinguish metrics where:

```text
lower is better
```

from metrics where:

```text
higher is better
```

Do not visually imply the wrong ranking.

Examples:

```text
MAE ↓
RMSE ↓
R² ↑
AUROC ↑
AUPRC ↑
```

When useful, include arrows in axis/table labels.

---

# 41. Baseline and ablation figures

For ablation analysis:

- keep evaluation protocol identical;
- change only the intended component;
- preserve consistent model colors;
- show uncertainty;
- clearly label the reference configuration.

Do not compare results generated using incompatible data splits without explicit disclosure.

---

# 42. Heatmaps

Heatmaps must include:

```text
colorbar
units/metric
clear row labels
clear column labels
```

If ranking matters, optionally annotate numeric values.

Do not use a color scale that visually exaggerates small differences without disclosing the range.

---

# 43. Feature importance

If showing feature importance, label the method:

```text
Gini importance
permutation importance
SHAP
coefficient magnitude
```

These methods have different interpretations.

Do not label all of them simply as:

```text
Feature importance
```

without methodological context.

---

# 44. SHAP figures

For SHAP:

- specify model;
- specify data split;
- specify background/reference population where relevant;
- avoid computing explanations from test data for model-development decisions;
- preserve feature names;
- document transformations.

SHAP values must not be described as causal effects.

---

# 45. Dimensionality-reduction plots

For PCA, t-SNE, UMAP:

Always label the method.

Do not interpret proximity as proof of mechanistic similarity.

For PCA, report explained variance when appropriate.

For t-SNE/UMAP, record important parameters and random seed.

---

# 46. Chemical structure figures

Chemical structures should:

- use consistent bond thickness;
- use consistent atom-label size;
- avoid overlapping annotations;
- preserve stereochemistry;
- avoid decorative effects;
- use vector output when possible.

Any structure highlighting must correspond to documented structural information.

---

# 47. Dataset split visualization

When visualizing:

```text
train
validation
test
```

use stable semantic colors throughout the project.

Recommended:

```text
Train        #0072B2
Validation   #E69F00
Test         #009E73
```

These colors should not be reused for unrelated semantic roles within the same figure.

---

# 48. Model-family colors

If a project contains multiple representation families, a recommended stable mapping is:

```text
Null/reference      Gray
PhysChem            Orange
RDKit2D             Bluish green
Morgan              Blue
Graph neural model  Purple
Fusion model        Vermillion
```

Once adopted within a project, do not change this mapping between figures.

---

# 49. Python plotting stack

Preferred core stack:

```text
Python
pandas
numpy
matplotlib
```

Optional where appropriate:

```text
scipy
statsmodels
scikit-learn
adjustText
```

Agents should avoid adding a plotting dependency unless it provides a clear scientific or technical benefit.

---

# 50. Matplotlib first

Publication plots should preferably be generated with:

```text
matplotlib
```

because it provides explicit control over:

- physical figure size;
- fonts;
- line widths;
- axes;
- vector export;
- panel layout;
- reproducibility.

---

# 51. Central plotting style module

Do not redefine plotting style independently in every script.

Recommended:

```text
src/
└── plotting/
    ├── style.py
    ├── colors.py
    └── utils.py
```

Example responsibilities:

```text
colors.py
→ project palettes and semantic colors

style.py
→ rcParams, fonts, line widths, figure defaults

utils.py
→ panel labels, save functions, overlap checks
```

---

# 52. Example central constants

Recommended concept:

```python
COLORS = {
    "primary": "#0072B2",
    "secondary": "#D55E00",
    "physchem": "#E69F00",
    "rdkit2d": "#009E73",
    "morgan": "#0072B2",
    "reference": "#7A7A7A",
}
```

Do not duplicate hex values across dozens of plotting scripts.

---

# 53. Standard save function

Implement one central save function.

Example behavior:

```text
save_figure(fig, output_stem)
```

should produce:

```text
.pdf
.svg
.png
```

with consistent:

- bounding box;
- transparency;
- DPI;
- metadata.

---

# 54. Script independence

Each final figure script should run from a clean environment using documented input paths.

Prefer:

```bash
python figures/scripts/fig_baseline_comparison.py
```

over notebook-only manual workflows.

Notebooks may be used for exploration, but publication figures should preferably have a standalone script.

---

# 55. Reproducibility

Any random visual operation must use a fixed seed.

Examples:

```text
jitter
bootstrap CI
sampled labels
UMAP
t-SNE
```

Record the seed.

---

# 56. Data aggregation

Never aggregate silently.

If plotting:

```text
mean MAE
```

retain the individual run-level values in the plotting data or source files.

Prefer:

```text
raw runs → aggregation script → figure
```

rather than storing only the final mean.

---

# 57. Numerical precision

Do not display meaningless precision.

Examples:

```text
MAE = 0.284
```

may be preferable to:

```text
MAE = 0.283947261
```

But calculations and source files must retain full numerical precision.

Rounding is presentation only.

---

# 58. Scientific notation

Use scientific notation consistently for very small or large values.

Example:

```text
1.2 × 10⁻⁶
```

rather than inconsistent mixtures of decimal and exponential notation.

---

# 59. Units

Axis labels must include units where applicable.

Preferred:

```text
Time (h)
Concentration (mg/L)
Clearance (L/h)
Papp (cm/s)
```

For dimensionless transformed targets:

```text
log10(Papp / (1 cm s⁻¹))
```

or the project's formally defined notation.

---

# 60. Abbreviations

Avoid unexplained abbreviations in figures.

Common project-specific abbreviations may be used only if:

- defined in the figure caption;
- defined consistently in the manuscript;
- unambiguous.

---

# 61. Figure file naming

Recommended:

```text
fig_<number>_<short_description>
```

Examples:

```text
fig_01_dataset_distribution
fig_02_baseline_validation
fig_03_predicted_vs_observed
fig_04_residual_analysis
fig_05_ablation
```

Supplementary:

```text
fig_s01_...
fig_s02_...
```

---

# 62. Do not overwrite final figures silently

If a figure script changes materially:

- preserve version control;
- update Git history;
- regenerate manifest;
- update source-data hash.

Do not manually edit the exported PDF/PNG without updating the source script.

---

# 63. Exploratory vs publication figures

Agents should distinguish:

## Exploratory

Purpose:

```text
analysis / diagnosis
```

Can contain:

- more annotations;
- diagnostic panels;
- temporary titles;
- intermediate statistics.

## Publication

Purpose:

```text
manuscript / final report
```

Must pass all publication QC rules.

Do not automatically promote an exploratory figure into a publication figure without review.

---

# 64. Figure caption support

For every final figure, agents should provide enough metadata to draft a caption containing:

- what is shown;
- dataset;
- groups/models;
- sample size;
- metric;
- meaning of error bars;
- statistical test if applicable;
- important transformations.

Do not embed an excessively long methodology paragraph inside the figure itself.

---

# 65. Journal-specific override

This file defines project defaults.

When a target journal is selected:

```text
TARGET JOURNAL REQUIREMENTS > AGENTS_Figure defaults
```

Agents must check:

- accepted file formats;
- maximum dimensions;
- minimum resolution;
- font requirements;
- color mode;
- panel-label conventions;
- supplementary-figure requirements.

Do not assume one journal's technical requirements apply universally.

---

# 66. Visual QC before acceptance

Every final figure must be visually inspected.

Check:

```text
□ labels readable
□ no overlapping text
□ no clipped labels
□ no cropped legend
□ no hidden data points
□ correct colors
□ correct units
□ correct axis labels
□ correct sample counts
□ correct error bars
□ correct statistical annotations
□ no unexpected blank regions
□ no rendering errors
□ no rasterized text when vector output is expected
```

---

# 67. Scientific QC before acceptance

Check:

```text
□ source files exist
□ plotted sample count matches source
□ data filtering documented
□ transformations documented
□ no fabricated values
□ metric recalculation matches stored results
□ aggregation is reproducible
□ uncertainty definition is correct
□ test/validation/split meanings are correct
□ no inconsistent comparison protocols
□ figure interpretation is supported by data
```

---

# 68. Automated QC

Where practical, plotting scripts should assert critical assumptions.

Examples:

```python
assert len(df) > 0
assert df["MAE"].notna().all()
assert set(df["split_seed"]) == {1, 2, 3, 4, 5}
```

For expected test predictions:

```python
assert len(predictions) == expected_n
```

Fail loudly if critical conditions are violated.

---

# 69. No silent exception handling

Avoid:

```python
try:
    ...
except:
    pass
```

in scientific plotting pipelines.

A missing or malformed input should produce:

```text
clear error
+
source path
+
reason
```

rather than an incomplete figure that appears valid.

---

# 70. Recommended figure workflow for agents

```text
Identify scientific question
        ↓
Locate authoritative source data
        ↓
Audit source data
        ↓
Create plotting table
        ↓
Record transformations/statistics
        ↓
Choose appropriate plot type
        ↓
Apply project plotting style
        ↓
Generate draft
        ↓
Visual QC
        ↓
Scientific QC
        ↓
Generate PDF/SVG/PNG
        ↓
Generate/update manifest
```

---

# 71. Machine-learning project figure set

For regression projects, the standard candidate figure library includes:

```text
Dataset target distribution
Baseline/model performance comparison
Validation stability
Predicted vs observed
Residual distribution
Residual vs predicted
Performance by target range
Error distribution
Representation ablation
Algorithm ablation
Feature fusion ablation
Feature importance / SHAP
Applicability-domain analysis
Uncertainty analysis
```

Not every project requires every figure.

Figures must answer a scientific question rather than merely fill a checklist.

---

# 72. Caco-2 / ADMET example

For a Caco-2 regression project, useful figure candidates may include:

```text
Fig. 1  Dataset and target distribution
Fig. 2  Core baseline comparison
Fig. 3  Representation × algorithm heatmap
Fig. 4  Predicted vs observed Caco-2 permeability
Fig. 5  Residual analysis
Fig. 6  Performance across permeability ranges
Fig. 7  Representation ablation
Fig. 8  Advanced-model comparison
```

These figures may only be produced when corresponding real results exist.

Do not create empty or simulated placeholder results and present them as experiments.

---

# 73. Figure acceptance gate

A figure is considered publication-ready only when:

```text
DATA INTEGRITY      PASS
SCIENTIFIC QC       PASS
VISUAL QC           PASS
REPRODUCIBILITY     PASS
OUTPUT FORMAT       PASS
```

Then mark:

```text
FIGURE STATUS = PUBLICATION_READY
```

Otherwise retain:

```text
FIGURE STATUS = DRAFT
```

---

# 74. Agent behavior when uncertain

If an agent is uncertain about:

- data meaning;
- units;
- sample identity;
- metric definition;
- statistical test;
- error-bar meaning;
- transformation;
- comparison protocol;

the agent must not guess.

Preferred behavior:

```text
record uncertainty
+
produce only scientifically supported components
+
flag the unresolved item
```

---

# 75. Core principle

Every scientific figure must satisfy:

```text
The figure should make the true data easier to understand,
not make the desired conclusion easier to believe.
```

---
# 76
Line width: 1.2–1.5 pt
Axis spine: 0.8–1.0 pt
Marker size: 4–6 pt
Error-bar capsize: 2–3 pt

---
# 77
Raw points alpha: 0.5–0.8
Confidence interval alpha: 0.15–0.25
Do not use transparency to hide outliers.

---
# 78
MAE / RMSE: usually 3 decimals
R²: 2–3 decimals
Pearson/Spearman: 2–3 decimals
p-value: report exact value when practical

---
# 79
Publication figures must be generated from scripts.
Screenshots of plots, notebook outputs, or GUI windows must not be used as final manuscript figures.

