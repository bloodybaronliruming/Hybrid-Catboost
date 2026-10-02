# Scientific figure specification

This specification applies to the study figures and their reproduction. The
figure captions describe the frozen study; regenerated figures must report their
own results and must not imply that reproduction has already been verified.

## Scientific content

- Derive every plotted value from the recorded data, predictions or metrics.
  Preserve row order, model identifiers, seed identities and label scale.
- Report all five runs. Distinguish sample standard deviation, population
  standard deviation, confidence intervals and prediction intervals.
- Define the sign of paired differences and identify the baseline. Differences
  from overlapping splits are descriptive unless a justified analysis is supplied.
- Keep training, validation and fixed-test results clearly labeled. Define
  diagnostic bins using the specified training subset.
- Include units or the original TDC logS scale on axes. Observed-versus-predicted
  plots use comparable axes and a clearly identified identity line.
- Do not smooth, clip, rescale or selectively omit observations to improve the
  appearance of results. Explain any necessary exclusion or display limit.
- Avoid unsupported significance annotations, invented results and claims of
  external validation or validated PBPK performance.

## Presentation

- Use a consistent layout, typography, line widths and model colors across
  figures. Labels and legends must remain legible at the intended print size.
- Use a colorblind-accessible palette with sufficient contrast; add line styles,
  markers or direct labels where color alone would be ambiguous.
- Keep backgrounds uncluttered. Use restrained grid lines and avoid decorative
  gradients, three-dimensional effects and unnecessary panels.
- Use sequential colors for ordered magnitudes and diverging colors only for
  quantities with a meaningful center. Label color bars.
- Explain abbreviations, panels, aggregation, sample counts and error bars in
  captions. Keep captions outside the figure image.

## Export and review

- Provide vector PDF or SVG where appropriate and high-resolution PNG exports.
  Raster figures must meet the submission requirement of at least 300 dpi at
  their intended display size; use higher resolution for detailed line art.
- Embed fonts and preserve mathematical symbols. Inspect exported figures for
  clipping, overlapping labels, broken symbols and loss of detail.
- Retain plotting code and source hashes so figure values remain traceable.
  Generated figure metadata records this specification's SHA-256.
- Verify displayed values against numerical tables and full-precision records.
  Follow a target journal's requirements when they are more specific.
