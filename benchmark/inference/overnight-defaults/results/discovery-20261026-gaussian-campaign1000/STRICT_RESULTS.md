# Complete-dataset inference summaries

EMPIRICAL. Every requested target must be finite, unique and converged before a dataset contributes. Missing/failed fits count against reliability. Per-slot results use fixed requested target roles; physical columns vary in random-support designs. The older per-target-by-column file mixes roles in those designs and is superseded. All Monte Carlo errors use independent datasets.

|Case|Method|Class|Valid/planned|Coverage ± MCSE|Rejection ± MCSE|Length|
|---|---|---|---|---|---|---|
|gaussian-benign|gaussian_default_hc3|active|1000/1000|0.9444 ± 0.0035044144904564185|1.0000 ± 0.0|0.2189|
|gaussian-benign|gaussian_default_hc3|null|1000/1000|0.9584 ± 0.0028909856100987077|0.0416 ± 0.002890985610098708|0.2198|
|gaussian-correlated|gaussian_default_hc3|active|1000/1000|0.9648 ± 0.0027070811381058172|0.1526 ± 0.005732896833722557|1.9648|
|gaussian-correlated|gaussian_default_hc3|null|1000/1000|0.9576 ± 0.0029750783336058716|0.0424 ± 0.002975078333605872|2.0611|
|gaussian-heavy|gaussian_default_hc3|active|1000/1000|0.9610 ± 0.0027652570670096198|0.9890 ± 0.0018336745519209132|0.3947|
|gaussian-heavy|gaussian_default_hc3|null|1000/1000|0.9648 ± 0.0026922496556241226|0.0352 ± 0.002692249655624123|0.3857|
|gaussian-hetero|gaussian_default_hc3|active|1000/1000|0.9662 ± 0.0026428288905157183|0.9980 ± 0.0006296001877353566|0.4021|
|gaussian-hetero|gaussian_default_hc3|null|1000/1000|0.9626 ± 0.0027728917989941343|0.0374 ± 0.0027728917989941347|0.3973|
|gaussian-hetero-target|gaussian_default_hc3|active|1000/1000|0.9664 ± 0.00265293763177314|0.9966 ± 0.0008179908979378125|0.4246|
|gaussian-hetero-target|gaussian_default_hc3|null|1000/1000|0.9634 ± 0.0027835567513862882|0.0366 ± 0.0027835567513862887|0.4106|
|gaussian-independent|gaussian_default_hc3|active|1000/1000|0.9644 ± 0.002641896973695155|0.9904 ± 0.0013819369557797306|0.4664|
|gaussian-independent|gaussian_default_hc3|null|1000/1000|0.9736 ± 0.0022153888931625404|0.0264 ± 0.002215388893162541|0.4627|
|gaussian-largen|gaussian_default_hc3|active|1000/1000|0.9544 ± 0.002954550620546104|1.0000 ± 0.0|0.2336|
|gaussian-largen|gaussian_default_hc3|null|1000/1000|0.9576 ± 0.00294803848871261|0.0424 ± 0.0029480384887126104|0.2316|
|gaussian-moderate|gaussian_default_hc3|active|1000/1000|0.9618 ± 0.002930759176279987|0.9584 ± 0.003039519481542877|0.5213|
|gaussian-moderate|gaussian_default_hc3|null|1000/1000|0.9732 ± 0.0023167060942482815|0.0268 ± 0.002316706094248282|0.5177|
|gaussian-random|gaussian_default_hc3|active|1000/1000|0.9690 ± 0.002458667331105419|0.9786 ± 0.002094379723552161|0.5046|
|gaussian-random|gaussian_default_hc3|null|1000/1000|0.9702 ± 0.0024572338102016255|0.0298 ± 0.002457233810201626|0.4973|
|gaussian-weak|gaussian_default_hc3|active|1000/1000|0.9584 ± 0.0028350445283059447|0.3282 ± 0.006509003543333428|0.4924|
|gaussian-weak|gaussian_default_hc3|null|1000/1000|0.9598 ± 0.002862193590963433|0.0402 ± 0.0028621935909634334|0.4857|
