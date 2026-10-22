# Iterators over the coordinates visited in one pass of coordinate descent.
#
# With fullPass = true they produce the candidate coordinates: all of
# 1:numCoordinates(iterate), unless setCandidates! restricted them (screening);
# otherwise only the coordinates stored in the sparse iterate (the active set).
# reset! switches between the two and, for RandomIterator, draws a new order.
abstract type AtomIterator end

Base.eltype(::AtomIterator) = Int64
Base.length(x::AtomIterator) = x.fullPass ? _numCandidates(x) : nnz(x.iterate)

_numCandidates(x::AtomIterator) = x.candidates === nothing ? ProximalBase.numCoordinates(x.iterate) : length(x.candidates)
_candidate(x::AtomIterator, i::Int) = x.candidates === nothing ? i : x.candidates[i]

# full passes visit only the coordinates in S, or all of them for nothing
function setCandidates!(x::AtomIterator, S::Union{Nothing, Vector{Int}})
  x.candidates = S
  x
end


# T = SparseIterate, SymmetricSparseIterate, AtomIterate
# last argument represents full pass
mutable struct OrderedIterator{T} <: AtomIterator
  iterate::T
  fullPass::Bool
  candidates::Union{Nothing, Vector{Int}}

  OrderedIterator{T}(iterate::Union{SparseIterate,SymmetricSparseIterate,AtomIterate}, fullPass) where {T} =
    new(iterate, fullPass, nothing)
end

OrderedIterator(iterate) = OrderedIterator{typeof(iterate)}(iterate, true)

Base.iterate(x::OrderedIterator) = iterate(x, 1)
function Base.iterate(x::OrderedIterator, i)
    # check if done
    done = x.fullPass ? i > _numCandidates(x) : i > nnz(x.iterate)
    if done
        return nothing
    end
    x.fullPass ? (_candidate(x, i), i + 1) : (x.iterate.nzval2ind[i], i + 1)
end

function reset!(x::OrderedIterator, fullPass::Bool)
  x.fullPass = fullPass
  x
end



mutable struct RandomIterator{T} <: AtomIterator
  iterate::T
  order::Vector{Int64}
  fullPass::Bool
  candidates::Union{Nothing, Vector{Int}}

  RandomIterator{T}(iterate::Union{SparseIterate,SymmetricSparseIterate,AtomIterate}, order, fullPass) where {T} =
    new(iterate, order, fullPass, nothing)
end

RandomIterator(iterate::Union{SparseIterate,SymmetricSparseIterate,AtomIterate}) =
  RandomIterator{typeof(iterate)}(iterate, collect(1:ProximalBase.numCoordinates(iterate)), true)

# draws a new uniformly random order (Fisher–Yates shuffle) over the
# coordinates of the next pass
function reset!(x::RandomIterator, fullPass::Bool)
  x.fullPass = fullPass
  newLength = fullPass ? _numCandidates(x) : nnz(x.iterate)
  @inbounds for i=1:newLength
    x.order[i] = i
  end
  @inbounds for i=1:newLength-1
    j = rand(i:newLength)
    x.order[i], x.order[j] = x.order[j], x.order[i]
  end
  x
end


Base.iterate(x::RandomIterator) = iterate(x, 1)
function Base.iterate(x::RandomIterator, i)
    # check if done
    done = x.fullPass ? i > _numCandidates(x) : i > nnz(x.iterate)
    if done
        return nothing
    end
    x.fullPass ? (_candidate(x, x.order[i]), i + 1) : (x.iterate.nzval2ind[x.order[i]], i + 1)
end
