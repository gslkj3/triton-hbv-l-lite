# P3L-0195 memory

- P185 single-anchor delta did not improve final material recovery.
- P194 rejected P193 because all five selected responses were constants.
- Learn only within-domain factor preference from all informative pairs.
- Cross-domain comparison remains the unchanged P167 point-time response.

## Executed result

- Status: `P195_ALL_PAIR_PREFERENCE_CLOSED_NEGATIVE`.
- Every selected response is state-dependent and pairwise Brier is below the
  constant 0.25 baseline.
- Domain-head oracle improved 55/73→56/73, but final material recovery fell
  40/73→39/73 under the unchanged cross-domain time response.
- Close all-pair preference; do not tune pair weights or add local models.

