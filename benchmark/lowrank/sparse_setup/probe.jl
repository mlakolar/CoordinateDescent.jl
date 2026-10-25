# Loaded only after production timing and profiles. Replay the exact source
# equations with state/count capture, and compare replay coefficients/iterations
# with saved production results. The measured methods are never instrumented.
const HDM102_STATE=Ref{Any}(nothing)
const HDM102_STATES=Any[]
const HDM102_COUNTS=Dict(:cd_sweeps=>0,:fista_steps=>0,:kkt_calls=>0)
function install_probe!()
  source=read(joinpath(dirname(pathof(HighDimensionalModels)),"lowrank.jl"),String)
  function section(start_marker,stop_marker)
    start=findfirst(start_marker,source);stop=findnext(stop_marker,source,last(start)+1)
    source[first(start):first(stop)-1]
  end
  kkt=section("function _lr_group_kkt(","# Exact row coordinate descent")
  kkt=replace(kkt,"  err = zero(eltype(A))"=>"  Main.HDM102_COUNTS[:kkt_calls] += 1\n  err = zero(eltype(A))")
  cd=section("function _lr_group!(","function _lr_group_fista!(")
  cd=replace(cd,"  for sweep in 1:max_iter"=>"  for sweep in 1:max_iter\n    Main.HDM102_COUNTS[:cd_sweeps] += 1")
  fista=section("function _lr_group_fista!(","function _lr_sparse")
  fista=replace(fista,"  for k in 1:max_iter"=>"  for k in 1:max_iter\n    Main.HDM102_COUNTS[:fista_steps] += 1")
  sparse=section("function _lr_sparse(","function _lr_weights(")
  marker=occursin("  rows = _activerows(B)",sparse) ? "  rows = _activerows(B)" : "  objective = sum(abs2, d.Y - d.X * B)"
  occursin(marker,sparse) || error("probe source marker changed")
  sparse=replace(sparse,marker=>"  Main.HDM102_STATE[] = (;A=copy(A),V=copy(V),w=copy(w),L)\n  push!(Main.HDM102_STATES, Main.HDM102_STATE[])\n"*marker)
  Base.include_string(HighDimensionalModels,kkt*cd*fista*sparse,"untimed-factor-probe.jl")
end
