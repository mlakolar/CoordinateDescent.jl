lib <- Sys.getenv('SELECTION_RLIB')
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(lib, .libPaths()))
needed <- c('SLOPE', 'knockoff', 'stabs')
missing <- needed[!vapply(needed,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing)) install.packages(missing,lib=lib,repos='https://cloud.r-project.org',Ncpus=2)
stopifnot(all(vapply(needed,requireNamespace,logical(1),quietly=TRUE)))
print(sapply(needed, function(x) as.character(packageVersion(x))))
