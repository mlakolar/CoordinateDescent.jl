module HD

export SoftThreshold, lasso

# soft-thersholding operator
# sign(z)(|z|-λ)_+
function SoftThreshold(z, lambda)
    abs(z) < lambda ? 0. : z > 0 ? z - lambda : z + lambda
end

######################################################################
#
#  Lasso Functions
#
######################################################################

# helper function for Active Shooting implementation of Lasso
# iterates over the active set
#
# TODO: add logging capabilities
function updateBeta!(beta, XtX, Xy, lambda; maxIter=1000, optTol=1e-7)

  nzval = beta.nzval
  rowval = beta.rowval

  iter = 1
  while iter <= maxIter
    fDone = true
    for j = 1:length(rowval)
      ci = rowval[j]
      # Compute the Shoot and Update the variable
      S0 = 0
      for k=1:length(rowval)
        S0 += XtX[rowval[k],ci] * nzval[k]
      end
      S0 = S0 - XtX[ci,ci]*nzval[j] - Xy[ci]
      oldVal = nzval[j]
      nzval[j] = SoftThreshold(-S0 / XtX[ci,ci], lambda[ci] / XtX[ci,ci])
      if abs(oldVal - nzval[j]) > optTol
        fDone = false
      end
    end

    iter = iter + 1
    if fDone
      break
    end
  end

  sparse(beta)
end

# finds index to add to the active_set
function addActiveSet!(active_set, beta, XtX, Xy, lambda)
  p = size(XtX, 1)
  nzval = beta.nzval
  rowval = beta.rowval

  val = 0
  ind = 0
  for j = setdiff([1:p], active_set)
    S0 = -Xy[j]
    for k = 1:length(rowval)
      S0 += XtX[rowval[k], j]*nzval[k]
    end
    if abs(S0) > lambda[j]
      if abs(S0) > val
        val = abs(S0);
        ind = j;
      end
    end
  end
  if ind != 0
    push!(active_set, ind)
    beta[ind] = eps()
  end
  return ind
end

function lasso(X, y, lambda;
               maxIter=2000, maxInnerIter=1000,
               optTol=1e-7, beta=[],
               XtX=[], Xy = [])

  if isempty(XtX) && isempty(Xy)
    n, p = size(X)

    XtX = (X'*X) ./ n
    Xy = (X'*y) ./ n
  else
    p = size(XtX, 2)
  end

  if isempty(beta)
    beta = sparsevec(zeros(p))
    active_set = Array(Integer, 0)
    ind = addActiveSet!(active_set, beta, XtX, Xy, lambda)
    if ind == 0
      return beta
    end
  else
    active_set = find(beta)
    beta = sparse(beta)
  end

  iter = 1;
  while iter < maxIter

    old_active_set = copy(active_set)
    updateBeta!(beta, XtX, Xy, lambda; maxIter=maxInnerIter, optTol=optTol)
    active_set = find(beta)
    addActiveSet!(active_set, beta, XtX, Xy, lambda)

    iter = iter + 1;
    if old_active_set == active_set
      break
    end
  end

  sparse(beta)
end


######################################################################
#
#  Group Lasso Functions
#
######################################################################

function findNonZeroGroups(beta, groups)
  active_set = []
  for gInd = 1 : length(groups)
    if norm(beta[groups[gInd]]) > 0
      push!(active_set, gInd)
    end
  end
  return active_set
end

function addGroupActiveSet!(active_set, beta, XtX, Xy, groups, lambda)

  numGroups = length(groups)

  val = 0
  ind = 0
  for j = setdiff([1:numGroups], active_set)
    S0 = -Xy[j]
    for k = 1:length(rowval)
      S0 += XtX[rowval[k], j]*nzval[k]
    end
    if abs(S0) > lambda[j]
      if abs(S0) > val
        val = abs(S0);
        ind = j;
      end
    end
  end
  if ind != 0
    push!(active_set, ind)
    beta[ind] = eps()
  end
  return ind

end

function group_lasso(X, y, groups, lambda;
                     maxIter=2000, maxInnerIter=1000,
                     optTol=1e-7, beta=[],
                     XtX=[], Xy = [])

  if isempty(XtX) && isempty(Xy)
    (n, p) = size(X)
    XtX = (X'*X) ./ n
    Xy = (X'*y) ./ n
  else
    p = size(XtX)
  end

  if isempty(beta)
    beta = zeros(p)
    active_set = Array(Integer, 0)
    ind = addGroupActiveSet!(active_set, beta, XtX, Xy, groups, lambda)
    if ind == 0
      return beta
    end
  else
    active_set = findNonZeroGroups(beta, groups)
  end

  iter = 1;
  while iter < maxIter

    old_active_set = copy(active_set)
    updateGroupBeta!(beta, XtX, Xy, groups, lambda; maxIter=maxInnerIter, optTol=optTol)
    active_set = findNonZeroGroups(beta)
    addGroupActiveSet!(active_set, beta, XtX, Xy, groups, lambda)

    iter = iter + 1;
    if old_active_set == active_set
      break
    end
  end

  sparse(beta)
end



end


