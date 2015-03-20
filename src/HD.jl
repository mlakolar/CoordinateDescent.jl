module HD

export SoftThreshold, lasso, prox_l2!

# soft-thersholding operator
# sign(z)(|z|-λ)_+
function SoftThreshold(z::Float64, lambda::Float64)
    abs(z) < lambda ? zero(z) : z > 0. ? z - lambda : z + lambda
end

######################################################################
#
#
######################################################################

# helper function for Active Shooting implementation of Lasso
# iterates over the active set
#
# TODO: add logging capabilities
function updateBeta!(beta, XX, Xy, lambda; maxIter=1000, optTol=1e-7)

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
        S0 += XX[rowval[k],ci] * nzval[k]
      end
      S0 = S0 - XX[ci,ci]*nzval[j] - Xy[ci]
      oldVal = nzval[j]
      nzval[j] = SoftThreshold(-S0 / XX[ci,ci], lambda[ci] / XX[ci,ci])
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
function addActiveSet!(active_set, beta, XX, Xy, lambda)
  p = size(XX, 1)
  nzval = beta.nzval
  rowval = beta.rowval

  val = 0
  ind = 0
  for j = setdiff([1:p], active_set)
    S0 = -Xy[j]
    for k = 1:length(rowval)
      S0 += XX[rowval[k], j]*nzval[k]
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
               XX=[], Xy = [])

  if isempty(XX) && isempty(Xy)
    n, p = size(X)

    XX = (X'*X) ./ n
    Xy = (X'*y) ./ n
  else
    p = size(XX, 2)
  end

  if isempty(beta)
    beta = sparsevec(zeros(p))
    active_set = Array(Integer, 0)
    ind = addActiveSet!(active_set, beta, XX, Xy, lambda)
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
    updateBeta!(beta, XX, Xy, lambda; maxIter=maxInnerIter, optTol=optTol)
    active_set = find(beta)
    addActiveSet!(active_set, beta, XX, Xy, lambda)

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

function findNonZeroGroups(beta::Array{Float64, 1}, groups::Array{Array{Int64, 1}, 1})
  active_set = Array(Int64, 0)
  for gInd = 1 : length(groups)
    if norm(beta[groups[gInd]]) > 0
      push!(active_set, gInd)
    end
  end
  return active_set
end

function addGroupActiveSet!(active_set::Array{Int64, 1},
                            XX::Array{Float64, 2}, Xy::Array{Float64, 1}, beta::Array{Float64, 1},
                            groups::Array{Array{Int64, 1}, 1}, lambda::Array{Float64, 1})

  numGroups = length(groups)
  largestG = maximum(map(length, groups))
  S0 = zeros(largestG)

  val = 0
  ind = 0
  for k = setdiff([1:numGroups], active_set)
    kGroup = groups[k]
    compute_group_residual!(S0, XX, Xy, beta, groups, active_set, k)
    normG = norm(S0[1:length(kGroup)])
    if normG > lambda[k]
      if normG > val
        val = normG
        ind = k
      end
    end
  end

  if ind != 0
    push!(active_set, ind::Int64)
  end

  return ind
end

# computes argmin lambda * |hat_x|_2 + |x-hat_x|^2 / 2
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
function minimize_one_group!(beta::Array{Float64, 1},
                             XX::Array{Float64, 2}, Xy::Array{Float64, 1},
                             lambda::Float64;
                             maxIter::Integer=1000, optTol::Float64 = 1e-7)

  z = copy(beta)
  new_beta = copy(beta)

  t = 1. / eigmax(XX)
  iter = 1.
  while iter < maxIter
    A_mul_B!(z, XX, beta)
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

function minimize_one_group_raw!(beta::Array{Float64, 1},
                             X::Array{Float64, 2}, Y::Array{Float64, 1},
                             lambda::Float64;
                             maxIter::Integer=1000, optTol::Float64 = 1e-7)
  n = size(X, 1)
  XX = X' * X / n
  Xy = X' * Y / n
  minimize_one_group!(beta, XX, Xy, lambda; maxIter=maxIter, optTol=optTol)
  nothing
end




# computes  (X^T)_k Y - sum_{j in active_set} (X^T)_j X_k beta_k
function compute_group_residual!(res::Array{Float64, 1},
                                 XX::Array{Float64, 2}, Xy::Array{Float64, 1}, beta::Array{Float64, 1},
                                 groups::Array{Array{Int64, 1}, 1}, active_set::Array{Int64, 1}, k::Int64)

  kGroup = groups[k]
  lenK = length(kGroup)
  for i=1:lenK
    res[i] = Xy[kGroup[i]]
    for j=active_set
      if j == k
        continue
      end
      jGroup = groups[j]
      for l=1:length(jGroup)
        res[i] -= XX[kGroup[i], jGroup[l]] * beta[jGroup[l]]
      end
    end
  end
  nothing
end


# helper function for Active Shooting implementation of Group Lasso
# iterates over the active set
#
# TODO: add logging capabilities
function minimize_active_groups!(beta::Array{Float64, 1},
                                 XX::Array{Float64, 2}, Xy::Array{Float64, 1},
                                 groups::Array{Array{Int64, 1}, 1}, active_set::Array{Int64, 1}, lambda::Array{Float64, 1};
                                 maxIter::Integer=1000, optTol::Float64=1e-7)

  numGroups = length(groups)
  largestG = maximum(map(length, groups))
  Xr = zeros(largestG)

  iter = 1
  while iter <= maxIter
    fDone = true
    for k=active_set
      kGroup = groups[k]
      lenK = length(kGroup)
      # Compute the Shoot and Update the variable

      compute_group_residual!(Xr, XX, Xy, beta, groups, active_set, k)

      # check if group is zero
      normG = norm(Xr[1:lenK])
      if normG < lambda[k]
        tmp_beta = zeros(lenK)
      else
        # update group
        tmp_beta = beta[kGroup]
        minimize_one_group!(tmp_beta, XX[kGroup, kGroup], Xr, lambda[k]; maxIter=maxIter, optTol=optTol)
      end
      if maximum(abs(beta[kGroup] - tmp_beta)) > optTol
        fDone = false
      end
      beta[kGroup] = tmp_beta
    end
    iter = iter + 1
    if fDone
      break
    end
  end
  nothing
end


function group_lasso_raw!(beta, X, y, groups, lambda;
                          maxIter=2000, maxInnerIter=1000, optTol=1e-7)

  n = size(X, 1)
  XX = (X'*X) ./ n
  Xy = (X'*y) ./ n

  group_lasso!(beta, XX, Xy, groups, lambda;
               maxIter=maxIter, maxInnerIter=maxInnerIter, optTol=optTol)
  nothing
end

function group_lasso!(beta, XX, Xy, groups, lambda;
                     maxIter=2000, maxInnerIter=1000, optTol=1e-7)

  p = size(XX, 1)

  if maximum(abs(beta - zeros(Float64, p))) < optTol
    active_set = Array(Int64, 0)
    ind = addGroupActiveSet!(active_set, XX, Xy, beta, groups, lambda)
    if ind == 0
      fill!(beta, 0.)
      return
    end
  else
    active_set = findNonZeroGroups(beta, groups)
  end

  iter = 1;
  while iter < maxIter

    old_active_set = copy(active_set)
    minimize_active_groups!(beta, XX, Xy, groups, active_set, lambda;
                            maxIter=maxInnerIter, optTol=optTol)
    active_set = findNonZeroGroups(beta, groups)
    addGroupActiveSet!(active_set, XX, Xy, beta, groups, lambda)

    iter = iter + 1;
    if old_active_set == active_set
      break
    end
  end

  nothing
end



end

