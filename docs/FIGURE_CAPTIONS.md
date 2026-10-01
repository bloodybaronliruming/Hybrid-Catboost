# Figure captions

These describe the frozen study figures. Newly generated runs must be assessed
against their own metrics and report manifests; captions do not certify agreement.

**Figure 1.** Hybrid-CatBoost feature processing and prediction workflow. Morgan fingerprints and 210 initial RDKit two-dimensional descriptors were computed from the parsed molecular structures. For each official seed (1–5), descriptor medians and zero-variance filtering were fitted on that seed’s training subset, retaining 207 descriptors for seeds 1, 2, 4 and 5 and 206 for seed 3. After concatenation, only the retained Ipc descriptor was transformed as ln(1 + Ipc). Five separate numeric-feature CatBoost models (GPU, Plain boosting) were fitted on their respective training subsets and each predicted the same 1,997 fixed-test records. The diagram summarizes the fitted pipelines; it does not depict an averaged-prediction ensemble.

**Figure 2.** Fixed-test MAE for all ten groups, five final runs each. Colored shapes are individual run metrics; black diamonds and horizontal bars in panel a denote the mean and sample SD of five run-level MAEs. Panel b shows the complete finite ridge MAE range on its own logarithmic axis. The SD is not a confidence interval. Exact values, including ridge, are in Table S1.

**Figure 3.** Paired validation MAE differences for seven comparisons. Dots show the five-seed mean and horizontal bars the sample SD of matched-seed differences. The dashed vertical reference is zero. Panel a includes every comparison; panel b repeats five smaller-magnitude comparisons on a separate labeled scale. Splits overlap, so the bars do not represent independent-sample confidence intervals.

**Figure 4.** Hybrid-CatBoost on 1,997 fixed-test rows. Panel a compares each observed TDC logS label with the arithmetic mean of five separately fitted predictions; the dashed line is identity. Panel b plots the corresponding signed residual (prediction minus observation), with a zero reference. The mean prediction is used only for descriptive plotting; the benchmark metrics in Table 3 remain means of individual-run metrics.

**Figure 5.** Post-test target-range analysis for five selected nonlinear models. Q1–Q4 are defined separately from each seed's training-label quartiles; a fixed-test row can change bins between seeds. Points show mean bin-level MAE (panel a) or signed residual (panel b), and bars show sample SD across five seed-level bin metrics. Colors follow representation family and marker shapes distinguish learners. This analysis is descriptive, not a fitted calibration or a newly validated solubility-class boundary.

**Figure S1.** Each point is a separately trained run on the same 1,997 fixed-test records. Panel a shows nine ordinary-range groups. Panel b displays all five finite ridge MAEs on a logarithmic axis because the range spans many orders of magnitude. No run was clipped, discarded, replaced or used to alter the frozen model. Exact values are in Table S1.
