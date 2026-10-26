# Complete-dataset inference summaries

EMPIRICAL. Every requested target must be finite, unique and converged before a dataset contributes. Missing/failed fits count against reliability. Per-slot results use fixed requested target roles; physical columns vary in random-support designs. The older per-target-by-column file mixes roles in those designs and is superseded. All Monte Carlo errors use independent datasets.

|Case|Method|Class|Valid/planned|Coverage ± MCSE|Rejection ± MCSE|Length|
|---|---|---|---|---|---|---|
|gaussian-benign|partialled_adaptive_cvmin_one_step_hc3|active|500/500|0.9468 ± 0.004415712517144671|1.0000 ± 0.0|0.2064|
|gaussian-benign|partialled_adaptive_cvmin_one_step_hc3|null|500/500|0.9596 ± 0.0038943564325890985|0.0404 ± 0.0038943564325891|0.2070|
|gaussian-benign|partialled_adaptive_cvmin_score_hc3|active|500/500|0.9512 ± 0.004316756742921137|1.0000 ± 0.0|0.2084|
|gaussian-benign|partialled_adaptive_cvmin_score_hc3|null|500/500|0.9580 ± 0.003942472088974719|0.0420 ± 0.00394247208897472|0.2098|
|gaussian-correlated|partialled_adaptive_cvmin_one_step_hc3|active|500/500|0.9620 ± 0.0040229800816652|0.2076 ± 0.009595239300531746|1.4906|
|gaussian-correlated|partialled_adaptive_cvmin_one_step_hc3|null|500/500|0.9532 ± 0.00448773880692698|0.0468 ± 0.004487738806926981|1.5713|
|gaussian-correlated|partialled_adaptive_cvmin_score_hc3|active|500/500|0.9804 ± 0.0030016161311529482|0.1024 ± 0.005924413671808326|1.8373|
|gaussian-correlated|partialled_adaptive_cvmin_score_hc3|null|500/500|0.9880 ± 0.0022720749655983172|0.0120 ± 0.0022720749655983177|1.8506|
|gaussian-heavy|partialled_adaptive_cvmin_one_step_hc3|active|500/500|0.9560 ± 0.004079372815167824|0.9972 ± 0.0014382130975024718|0.3155|
|gaussian-heavy|partialled_adaptive_cvmin_one_step_hc3|null|500/500|0.9744 ± 0.003531887378854799|0.0256 ± 0.0035318873788548|0.3141|
|gaussian-heavy|partialled_adaptive_cvmin_score_hc3|active|500/500|0.9636 ± 0.0036354522450517896|0.9968 ± 0.0014912974002095434|0.3299|
|gaussian-heavy|partialled_adaptive_cvmin_score_hc3|null|500/500|0.9720 ± 0.0036740982561700745|0.0280 ± 0.0036740982561700745|0.3314|
|gaussian-hetero|partialled_adaptive_cvmin_one_step_hc3|active|500/500|0.9468 ± 0.004731203713343392|0.9996 ± 0.00039999999999999986|0.3296|
|gaussian-hetero|partialled_adaptive_cvmin_one_step_hc3|null|500/500|0.9608 ± 0.003857003104210804|0.0392 ± 0.003857003104210805|0.3240|
|gaussian-hetero|partialled_adaptive_cvmin_score_hc3|active|500/500|0.9532 ± 0.004305414403558935|0.9996 ± 0.00039999999999999986|0.3447|
|gaussian-hetero|partialled_adaptive_cvmin_score_hc3|null|500/500|0.9596 ± 0.004015960143196579|0.0404 ± 0.0040159601431965794|0.3417|
|gaussian-hetero-target|partialled_adaptive_cvmin_one_step_hc3|active|500/500|0.9448 ± 0.004836600604848578|0.9984 ± 0.0007975915650313875|0.3485|
|gaussian-hetero-target|partialled_adaptive_cvmin_one_step_hc3|null|500/500|0.9648 ± 0.0036808990798542877|0.0352 ± 0.003680899079854288|0.3360|
|gaussian-hetero-target|partialled_adaptive_cvmin_score_hc3|active|500/500|0.9572 ± 0.004277020670001869|0.9976 ± 0.0009748747414529958|0.3647|
|gaussian-hetero-target|partialled_adaptive_cvmin_score_hc3|null|500/500|0.9636 ± 0.0037226060214831168|0.0364 ± 0.0037226060214831176|0.3543|
|gaussian-independent|partialled_adaptive_cvmin_one_step_hc3|active|500/500|0.9348 ± 0.005187652984440454|0.9988 ± 0.000893709927700112|0.3548|
|gaussian-independent|partialled_adaptive_cvmin_one_step_hc3|null|500/500|0.9700 ± 0.003391608174896393|0.0300 ± 0.0033916081748963938|0.3511|
|gaussian-independent|partialled_adaptive_cvmin_score_hc3|active|500/500|0.9584 ± 0.004051136063725041|0.9984 ± 0.000978158262846227|0.3816|
|gaussian-independent|partialled_adaptive_cvmin_score_hc3|null|500/500|0.9692 ± 0.0034708520030756864|0.0308 ± 0.0034708520030756872|0.3830|
|gaussian-largen|partialled_adaptive_cvmin_one_step_hc3|active|500/500|0.9420 ± 0.004581919682398622|1.0000 ± 0.0|0.1912|
|gaussian-largen|partialled_adaptive_cvmin_one_step_hc3|null|500/500|0.9488 ± 0.0045517641760342165|0.0512 ± 0.004551764176034217|0.1891|
|gaussian-largen|partialled_adaptive_cvmin_score_hc3|active|500/500|0.9432 ± 0.004559506354463471|1.0000 ± 0.0|0.1930|
|gaussian-largen|partialled_adaptive_cvmin_score_hc3|null|500/500|0.9476 ± 0.004577158590383671|0.0524 ± 0.004577158590383672|0.1915|
|gaussian-moderate|partialled_adaptive_cvmin_one_step_hc3|active|500/500|0.9220 ± 0.005978585700142459|0.9920 ± 0.001754467529658286|0.3975|
|gaussian-moderate|partialled_adaptive_cvmin_one_step_hc3|null|500/500|0.9744 ± 0.00319839639178103|0.0256 ± 0.0031983963917810304|0.3918|
|gaussian-moderate|partialled_adaptive_cvmin_score_hc3|active|500/500|0.9500 ± 0.004630646366496051|0.9912 ± 0.001836263523423288|0.4267|
|gaussian-moderate|partialled_adaptive_cvmin_score_hc3|null|500/500|0.9748 ± 0.003229672051946799|0.0252 ± 0.0032296720519467996|0.4268|
|gaussian-random|partialled_adaptive_cvmin_one_step_hc3|active|500/500|0.9460 ± 0.004539737705735183|0.9968 ± 0.0011234073562943906|0.3840|
|gaussian-random|partialled_adaptive_cvmin_one_step_hc3|null|500/500|0.9668 ± 0.0038659294899989166|0.0332 ± 0.003865929489998917|0.3777|
|gaussian-random|partialled_adaptive_cvmin_score_hc3|active|500/500|0.9588 ± 0.003959352591761913|0.9952 ± 0.0013702829454902375|0.4122|
|gaussian-random|partialled_adaptive_cvmin_score_hc3|null|500/500|0.9672 ± 0.0038101646564707454|0.0328 ± 0.003810164656470746|0.4114|
|gaussian-weak|partialled_adaptive_cvmin_one_step_hc3|active|500/500|0.9556 ± 0.0042442085968088855|0.4668 ± 0.009636453833489263|0.3757|
|gaussian-weak|partialled_adaptive_cvmin_one_step_hc3|null|500/500|0.9644 ± 0.003738077908886572|0.0356 ± 0.0037380779088865726|0.3703|
|gaussian-weak|partialled_adaptive_cvmin_score_hc3|active|500/500|0.9616 ± 0.003914271713370468|0.4704 ± 0.009458679253809853|0.4027|
|gaussian-weak|partialled_adaptive_cvmin_score_hc3|null|500/500|0.9632 ± 0.0037788993318585025|0.0368 ± 0.003778899331858503|0.4032|
