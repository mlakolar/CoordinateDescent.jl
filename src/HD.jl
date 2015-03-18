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

function computeU!(new_theta, old_beta, grad, t, lambda)
  normG = 0
  new_theta[:] = old_beta - t .* grad
  normG = norm(new_theta, 2)
  tmp = max(1.-(t*lambda)/normG, 0.)
  new_theta *= tmp
end

function updateOneGroupSimple!(beta, X, Y, lambda; maxIter=1000, optTol=1e-7)
  n, sizeG = size(X)

  new_beta = beta[:]
  old_beta = beta[:]
  new_theta = beta[:]
  old_theta = beta[:]

  sXtX = X' * X / n
  sS0 = -X' * Y / n

  grad = zeros(Float64, sizeG)
  delta = zeros(Float64, sizeG)
  t = 1.
  iter = 1.
  while iter < maxIter
    if mod(iter, 1) == 0
      @show iter
    end
    old_beta[:] = new_beta[:]
    old_theta[:] = new_theta[:]
    # compute gradient
    grad[:] = sXtX * old_beta + sS0
    while true
      # computeU
      computeU!(new_theta, old_beta, grad, t, lambda)
      delta[:] = new_theta[:] - old_beta[:]
      @show normD = norm(delta)^2

      # computeLoss
      @show lNew = dot(new_theta, sXtX * new_theta) / 2. + dot(sS0, new_theta) + lambda * norm(new_theta)
      @show lOld = dot(old_beta, sXtX * old_beta) / 2. + dot(sS0, old_beta) + lambda * norm(old_beta)

      if lOld < lNew
        @show t = 0.8 * t
      else
        break
      end
    end

    new_beta[:] = old_theta[:] + iter / (iter + 3.) .* (new_theta[:] - old_theta[:])

    iter = iter + 1
    @show maximum(abs(new_beta - old_beta))
    if maximum(abs(new_beta - old_beta)) < optTol
      break
    end
  end

  beta[:] = new_beta[:]
end


function updateOneGroup!(beta, XtX, S0, rG, lambda; maxIter=1000, optTol=1e-7)
  sizeG = length(rG)

  new_beta = beta[rG]
  old_beta = beta[rG]
  new_theta = beta[rG]
  old_theta = beta[rG]

  sXtX = XtX[rG, rG]
  sS0 = S0[rG]

  grad = zeros(Float64, sizeG)
  delta = zeros(Float64, sizeG)
  t = 1.
  iter = 1.
  while iter < maxIter
    if mod(iter, 1) == 0
      @show iter
    end
    old_beta[:] = new_beta[:]
    old_theta[:] = new_theta[:]
    # compute gradient
    grad[:] = sXtX * old_beta + sS0
    while true
      # computeU
      computeU!(new_theta, old_beta, grad, t, lambda)
      delta[:] = new_theta[:] - old_beta[:]
      normD = norm(delta)^2

      # computeLoss
      lNew = dot(new_theta, sXtX * new_theta) / 2. + dot(sS0, new_theta)
      lOld = dot(old_beta, sXtX * old_beta) / 2. + dot(sS0, old_beta) + dot(grad, delta) + normD / (2. * t)

      if lOld < lNew
        t = 0.8 * t
      else
        break
      end
    end

    new_beta[:] = old_theta[:] + iter / (iter + 3.) .* (new_theta[:] - old_theta[:])

    iter = iter + 1
    if maximum(abs(new_beta - old_beta)) < optTol
      break
    end
  end

  beta[rG] = new_beta[:]
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

  iter = 1
  while iter <= maxIter
    fDone = true
    for j = active_set
      rG = groups[j]
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
        for ii=1:length(rG)
          for jj=1:length(cG)
            S0[ii] += XtX[rG[ii], cG[jj]]*beta[cG[jj]]
          end
        end
      end

      # store old value of beta
      for ii=1:length(rG)
        old_beta[ii] = beta[rG[ii]]
      end

      # check if group is zero
      normG = 0
      for ii=1:length(rG)
        normG += S0[ii]^2
      end
      normG = sqrt(normG)
      if normG < lambda[j]
        for ii=1:length(rG)
          beta[rG[ii]] = 0
        end
      end

      # update group
      updateOneGroup!(beta, XtX, S0, rG, lambda; maxIter=50, optTol=optTol)

      # if change is big, we are not done
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


