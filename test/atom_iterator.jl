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



end
