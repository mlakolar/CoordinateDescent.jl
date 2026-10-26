# Evidence artifacts

RESULTS.md gives the recommendation and scope. STATISTICS.md, RUNTIME.md,
ORDER-SENSITIVITY.md and OBJECTIVES.md retain the full tables and adverse
findings. CSVs provide Monte Carlo intervals, paired differences, zero-event
bounds, runtime/allocation summaries, RSS and original-objective checks.

Checked-in Markdown spacing and CSV line endings are normalized for review.
Archives retain the original byte output, source snapshots, environment
manifests, all reporting rows, provenance and failed attempts. Generated
matrices and serialized complete fitted objects remain on Discovery.

- reporting-inputs-v1.tar.gz: 7,121,835 bytes; SHA256 `5d79ba675b7dfaa7fbd11ba88c30815056be62eeeb9092cb69a917081ccd0c93`.
- worker-profiles-v1.tar.gz: 3,912,619 bytes; SHA256 `845fd17f98392517f5a3bd2f69179c8218b0d1130273bd9b5ea4a722db529d71`.

Both artifacts were byte-verified after transfer and opened as readable tar archives. The reporting bundle contains 3,370 files, including its checksum manifest; the profile bundle contains all 4,536 primary CPU/allocation reports. No numerical fitting or aggregation ran locally.
