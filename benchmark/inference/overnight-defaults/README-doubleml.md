# Official DoubleML logistic reference

Native DoubleMLLPLR at official immutable revision 6884b5f9816f180324d854a4d81b22e9d334f5e0. Source: https://github.com/DoubleML/doubleml-for-py/blob/6884b5f9816f180324d854a4d81b22e9d334f5e0/doubleml/plm/lplr.py . Both nuisance_space and instrument scores, five outer and five inner folds, one split repetition. Native score convergence failure raises an error. No hand changes to native estimator.

DoubleML requires explicit nuisance learners. Here ml_M is scikit-learn LogisticRegressionCV, ten default Cs, l1/liblinear, 5-fold log-loss CV, tolerance 1e-8, max_iter10000. Auxiliary ml_t/ml_m are LassoCV, 60 levels, 5 folds, tolerance1e-8, max_iter100000. All workers use one thread. These are explicit sparse-learner choices, not a package-wide default and not tuning matched to other methods. Native liblinear intercept/penalty conventions are retained. Convergence warnings are rejected and other warnings retained.

Matched saved datasets: ten original logistic designs and the two Liu class-conditional designs. Exactly TWO target roles per dataset: first originally requested target and last originally requested null. Belloni-null has two null roles. This is a focused cross-fitting diagnostic, not the ten-target comparison. Native normal intervals, native score sandwich independently recomputed; score mean and pvalue parity checked. All missing/failed targets count against complete-dataset availability.

The DoubleML documentation's make_lplr_LZZ2020 is a nonlinear ML DGP and is different from our class-conditional high-dimensional paper setting. Its existence does not establish coverage in our designs. Official docs: https://docs.doubleml.org/stable/guide/models.html .
