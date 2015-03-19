module HD

export SoftThreshold, lasso

# soft-thersholding operator
# sign(z)(|z|-λ)_+
function SoftThreshold(z, lambda)
    abs(z) < lambda ? zero(z) : z > 0 ? z - lambda : z + lambda
end

######################################################################
#
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
  largestG = maximum(map(length, groups))
  S0 = zeros(largestG)

  val = 0
  ind = 0
  for j = setdiff([1:numGroups], active_set)
    rG = groups[j]
    for ii = 1:length(rG)
      S0[ii] = -Xy[rG[ii]]
    end
    for k=active_set
      cG = groups[k]
      for ii=1:length(rG)
        for jj=1:length(cG)
          S0[ii] += XtX[rG[ii], cG[jj]]*beta[cG[jj]]
        end
      end
    end
    normG = norm(S0[1:length(rG)])
    if normG > lambda[j]
      if normG > val
        val = normG
        ind = j
      end
    end
  end

  if ind != 0
    push!(active_set, ind)
    rG = groups[ind]
    for ii = 1:length(rG)
      beta[rG[ii]] = eps()
    end
  end

  return ind
end

# computes argmin lambda * |hat_x\|_2 + |x-hat_x|^2 / 2
function prox_l2!(hat_x, x, lambda)
  tmp = max(1. - lambda / norm(x), 0)
  for i=1:length(x)
    hat_x[i] = tmp * x[i]
  end
  nothing
end

#TODO: implementing line search could be better
#TODO: acceleration could improve this
# computes argmin lambda * |beta|_2 + |y-X*beta|^2 / (2*n)
function minimize_one_group!(beta, X, y, lambda;
                              maxIter=100, optTol = 1e-7,
                              XtX=[], Xy = [])

  if isempty(XtX)
    n = size(X, 1)
    XtX = X' * X / n
    Xy = X' * y / n
  end
  z = copy(beta)
  new_beta = copy(beta)

  t = 1. / eigmax(XtX)
  iter = 1.
  while iter < maxIter
    A_mul_B!(z, XtX, beta)
    for i=1:length(beta)
      z[i]= beta[i] - t*(z[i]-Xy[i])
    end
    prox_l2!(new_beta, z, t * lambda)
    fDone = true
    for i=1:length(beta)
      if abs(new_beta[i]-beta[i]) > optTol
        fDone = false
      end
      beta[i] = new_beta[i]
    end
    if fDone
      break
    end
    iter = iter + 1
  end
  nothing
end



# helper function for Active Shooting implementation of Group Lasso
# iterates over the active set
#
# TODO: add logging capabilities
function updateGroupBeta!(beta, XtX, Xy, groups, active_set, lambda; maxIter=1000, optTol=1e-7)

  numGroups = length(groups)
  largestG = maximum(map(length, groups))
  S0 = zeros(largestG)
  old_beta = zeros(largestG)
  tmp_beta = zeros(largestG)

  iter = 1
  while iter <= maxIter
    fDone = true
    for j = active_set
      rG = groups[j]
      sizeG = length(rG)
      # Compute the Shoot and Update the variable

      # obtain the residual vector
      for ii = 1:length(rG)
        S0[ii] = -Xy[rG[ii]]
      end
      for k=active_set
        if j == k
          continue
        end
        cG = groups[k]
        for ii=1:sizeG
          for jj=1:sizeG
            S0[ii] += XtX[rG[ii], cG[jj]]*beta[cG[jj]]
          end
        end
      end

      # store old value of beta
      for ii=1:sizeG
        old_beta[ii] = beta[rG[ii]]
      end

      # check if group is zero
      normG = 0
      for ii=1:sizeG
        normG += S0[ii]^2
      end
      normG = sqrt(normG)
      if normG < lambda[j]
        fill!(tmp_beta, 0.)
      else
        # update group
        minimize_one_group!(sub(tmp_beta, 1:length(rG)), [], [], lambda[j]; maxIter=200, optTol=optTol,
                          XtX=sub(XtX, rG[1]:rG[end], rG[1]:rG[end]), Xy=sub(S0, 1:length(rG)))
      end
      for ii=1:length(rG)
        beta[rG[ii]] = tmp_beta[ii]
        if abs(old_beta[ii] - beta[rG[ii]]) > optTol
          fDone = false
        end
      end
    end

    iter = iter + 1
    if fDone
      break
    end
  end
  nothing
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
    updateGroupBeta!(beta, XtX, Xy, groups, active_set, lambda; maxIter=maxInnerIter, optTol=optTol)
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


