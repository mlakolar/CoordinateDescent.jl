These fixtures were generated under Slurm on Discovery by
`benchmark/structured/golden.R`, with R 4.5.3, genlasso 1.6.1 and glmgen
0.0.3 at commit `9e6061503d6dd103712a0c07f6be7293c1c58aaf`.
`provenance.txt` records R and dependency versions. Seeds are 8101–8103.
The four levels use `lambda_R = n * lambda_Julia`.

The genlasso fits are evaluated along its complete exact paths. Coefficients
and degrees of freedom cover fused regression, cyclic graph regression, and
trend orders 0–3 on unit and irregular locations. Glmgen supplies a second
trend-filtering reference. CI reads these fixed fixtures and needs no R
installation. Regenerate references only through the Discovery workflow.
