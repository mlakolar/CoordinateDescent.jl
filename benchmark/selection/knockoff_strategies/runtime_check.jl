using Serialization,Test, SHA
root=ENV["HDM_KNOCKOFF_RUNTIME"];rep=parse(Int,ARGS[1])
@testset "baseline candidate complete legacy fields" begin
 for case in 1:5,input in 1:3
  b=joinpath(root,"baseline-case-$case-rep-$rep");c=joinpath(root,"candidate-case-$case-rep-$rep")
  @test read(joinpath(b,"input-$input.bin"))==read(joinpath(c,"input-$input.bin"))
  @test isequal(deserialize(joinpath(b,"fields-$input-equicorrelated.bin")),deserialize(joinpath(c,"fields-$input-equicorrelated.bin")))
  @test isequal(deserialize(joinpath(b,"matrix-$input-equicorrelated.bin")),deserialize(joinpath(c,"matrix-$input-equicorrelated.bin")))
 end
end
