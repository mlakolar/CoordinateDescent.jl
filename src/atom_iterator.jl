abstract type AtomIterator end

Base.eltype(::AtomIterator) = Int64
Base.length(x::AtomIterator) = x.fullPass ? ProximalBase.numCoordinates(x.iterate) : nnz(x.iterate)


# T = SparseIterate, SymmetricSparseIterate, AtomIterate
# last argument represents full pass
mutable struct OrderedIterator{T} <: AtomIterator
  iterate::T
  fullPass::Bool

  OrderedIterator{T}(iterate::Union{SparseIterate,SymmetricSparseIterate,AtomIterate}, fullPass) where {T} = new(iterate, fullPass)
end

OrderedIterator(iterate) = OrderedIterator{typeof(iterate)}(iterate, true)

Base.iterate(x::OrderedIterator) = iterate(x, 1)
function Base.iterate(x::OrderedIterator, i)
    # check if done
    done = x.fullPass ? i > ProximalBase.numCoordinates(x.iterate) : i > nnz(x.iterate)
    if done
        return nothing
    end
    x.fullPass ? (i, i + 1) : (x.iterate.nzval2ind[i], i + 1)
end


# Base.start(x::OrderedIterator) = 1
# # current item, next state
# Base.next(x::OrderedIterator, i) = x.fullPass ? (i, i + 1) : (x.iterate.nzval2ind[i], i + 1)
# Base.done(x::OrderedIterator, i) = x.fullPass ? i > ProximalBase.numCoordinates(x.iterate) : i > nnz(x.iterate)
#
function reset!(x::OrderedIterator, fullPass::Bool)
  x.fullPass = fullPass
  x
end



mutable struct RandomIterator{T} <: AtomIterator
  iterate::T
  order::Vector{Int64}
  fullPass::Bool

  RandomIterator{T}(iterate::Union{SparseIterate,SymmetricSparseIterate,AtomIterate}, order, fullPass) where {T} =
    new(iterate, order, fullPass)
end

RandomIterator(iterate::Union{SparseIterate,SymmetricSparseIterate,AtomIterate}) =
  RandomIterator{typeof(iterate)}(iterate, collect(1:ProximalBase.numCoordinates(iterate)), true)

function reset!(x::RandomIterator, fullPass::Bool)
  x.fullPass = fullPass
  newLength = fullPass ? ProximalBase.numCoordinates(x.iterate) : nnz(x.iterate)
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
    done = x.fullPass ? i > ProximalBase.numCoordinates(x.iterate) : i > nnz(x.iterate)
    if done
        return nothing
    end
    x.fullPass ? (x.order[i], i + 1) : (x.iterate.nzval2ind[x.order[i]], i + 1)
end


# Base.start(x::RandomIterator) = 1
# Base.next(x::RandomIterator, i) = x.fullPass ? (x.order[i], i + 1) : (x.iterate.nzval2ind[x.order[i]], i + 1)
# Base.done(x::RandomIterator, i) = x.fullPass ? i > ProximalBase.numCoordinates(x.iterate) : i > nnz(x.iterate)
