|Case|Method|Valid / attempted|Mean seconds|Input SD|95% t interval|Max exact gap|Max smooth KKT|
|---|---|---:|---:|---:|---|---:|---:|
|small|conquer|6/6|0.03342|0.00248|[0.0273, 0.0396]|0.000267|6.85e-07|
|small|gurobi-primal-primal-warm|6/6|0.2286|0.006|[0.214, 0.244]|1.39e-12|nan|
|small|lp-dual-dual-warm|6/6|0.2182|0.00472|[0.206, 0.23]|1.39e-12|nan|
|small|newton-0.2|6/6|0.003403|0.000136|[0.00306, 0.00374]|0.000267|5.28e-06|
|small|quantreg-fn|6/6|0.2461|0.00603|[0.231, 0.261]|1.85e-12|nan|
|tall|conquer|6/6|0.2172|0.00876|[0.195, 0.239]|2.75e-05|4.43e-07|
|tall|gurobi-primal-primal-warm|0/6|nan|nan|[nan, nan]|1.38e-12|nan|
|tall|lp-dual-dual-warm|6/6|2.534|0.017|[2.49, 2.58]|1.88e-10|nan|
|tall|newton-0.2|6/6|0.0282|0.00102|[0.0257, 0.0307]|2.75e-05|4.85e-06|
|tall|quantreg-fn|6/6|3.086|0.0956|[2.85, 3.32]|1.26e-10|nan|
|tall100k|conquer|6/6|4.246|0.123|[3.94, 4.55]|2.71e-06|5e-07|
|tall100k|gurobi-primal-primal-warm|0/6|nan|nan|[nan, nan]|nan|nan|
|tall100k|lp-dual-dual-warm|0/6|nan|nan|[nan, nan]|2.26e-11|nan|
|tall100k|newton-0.2|6/6|0.4167|0.0376|[0.323, 0.51]|2.71e-06|4.4e-06|
|tall100k|quantreg-fn|6/6|41.52|1.41|[38, 45]|2.4e-11|nan|
