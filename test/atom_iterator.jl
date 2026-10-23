module AtomIteratorTest

using Test
using ProximalBase
using CoordinateDescent
using Random

Random.seed!(1)


@testset "Ordered" begin
    p = 5
    x = SparseIterate(p)
    x[2] = 1.
    x[1] = 2.

    it = CoordinateDescent.OrderedIterator(x)

    fullPass   = collect(1:5)
    sparsePass = [2, 1]

    @test collect(it) == fullPass           # this should be a full pass

    CoordinateDescent.reset!(it, true)
    @test collect(it) == fullPass           # this should be a full pass

    CoordinateDescent.reset!(it, false)
    @test collect(it) == sparsePass         # pass over non-zeros


    # SymmetricSparseIterate
    x = SymmetricSparseIterate(3)
    x[2] = 1.
    x[1] = 2.

    it = CoordinateDescent.OrderedIterator(x)

    fullPass   = collect(1:6)
    sparsePass = [2, 1]

    @test collect(it) == fullPass           # this should be a full pass

    CoordinateDescent.reset!(it, true)
    @test collect(it) == fullPass           # this should be a full pass

    CoordinateDescent.reset!(it, false)
    @test collect(it) == sparsePass         # pass over non-zeros
end

@testset "Random" begin
    p = 50
    s = 10

    x = SparseIterate(p)
    for i=1:s
      x[rand(1:p)] = randn()
    end

    it = CoordinateDescent.RandomIterator(x)

    @test collect(it) == collect(1:ProximalBase.numCoordinates(it.iterate))   # this should be a full pass over 1:p

    CoordinateDescent.reset!(it, true)
    @test collect(it) == it.order                       # this should be a full pass over 1:p in a random order

    CoordinateDescent.reset!(it, false)
    @test collect(it) == [x.nzval2ind[it.order[i]] for i=1:nnz(x)] # this should be a sparse pass

    # SymmetricSparseIterate
    x = SymmetricSparseIterate(10)
    for i=1:s
      x[rand(1:55)] = randn()
    end

    it = CoordinateDescent.RandomIterator(x)

    @test collect(it) == collect(1:ProximalBase.numCoordinates(it.iterate))   # this should be a full pass over 1:p

    CoordinateDescent.reset!(it, true)
    @test collect(it) == it.order                       # this should be a full pass over 1:p in a random order

    CoordinateDescent.reset!(it, false)
    @test collect(it) == [x.nzval2ind[it.order[i]] for i=1:nnz(x)] # this should be a sparse pass

end

@testset "candidates" begin
    # full passes visit only the candidate coordinates; active passes are unchanged
    x = SparseIterate(6)
    x[2] = 1.
    x[5] = 2.
    for it in (CoordinateDescent.OrderedIterator(x), CoordinateDescent.RandomIterator(x))
        CoordinateDescent.setCandidates!(it, [4, 2, 6])
        CoordinateDescent.reset!(it, true)
        @test sort(collect(it)) == [2, 4, 6]
        @test length(it) == 3
        CoordinateDescent.reset!(it, false)
        @test sort(collect(it)) == [2, 5]
        CoordinateDescent.setCandidates!(it, nothing)
        CoordinateDescent.reset!(it, true)
        @test sort(collect(it)) == 1:6
    end
    it = CoordinateDescent.OrderedIterator(x)
    CoordinateDescent.setCandidates!(it, [4, 2, 6])
    @test collect(it) == [4, 2, 6]
end

@testset "AtomIterate" begin
    # the coordinates are the groups, and active passes visit the active groups
    x = AtomIterate(10, [1:2, 3:5, 6:6, 7:10])
    x[7] = 1.
    x[4] = 2.
    for it in (CoordinateDescent.OrderedIterator(x), CoordinateDescent.RandomIterator(x))
        CoordinateDescent.reset!(it, true)
        @test sort(collect(it)) == 1:4
        @test length(it) == 4
        CoordinateDescent.reset!(it, false)
        @test sort(collect(it)) == [2, 4]
        @test length(it) == 2
    end
    it = CoordinateDescent.OrderedIterator(x)
    CoordinateDescent.reset!(it, false)
    @test collect(it) == [4, 2]           # in the order the groups became active
end

@testset "random order on the active set only" begin
    # order = :active: full passes in the order 1, 2, …, passes over the active
    # set in a new random order each time
    p = 50
    x = SparseIterate(p)
    for k in 1:2:p
      x[k] = randn()
    end
    it = CoordinateDescent.RandomIterator(x, false)
    orders = Set{Vector{Int}}()
    for pass in 1:10
      CoordinateDescent.reset!(it, true)
      @test collect(it) == 1:p
      CoordinateDescent.reset!(it, false)
      active = collect(it)
      @test sort(active) == 1:2:p
      push!(orders, active)
    end
    @test length(orders) > 1
    CoordinateDescent.setCandidates!(it, [9, 4, 7])
    CoordinateDescent.reset!(it, true)
    @test collect(it) == [9, 4, 7]                  # the candidates, in their order

    # the iterator that each order of CDOptions picks
    @test CoordinateDescent._passIterator(x, CDOptions(;order=:cyclic)) isa CoordinateDescent.OrderedIterator
    @test CoordinateDescent._passIterator(x, CDOptions(;order=:random)).shuffleFull
    @test !CoordinateDescent._passIterator(x, CDOptions(;order=:active)).shuffleFull
end

end
