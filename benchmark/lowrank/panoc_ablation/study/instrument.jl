function instrument_native!(ext)
 @eval ext const _HDM_STUDY_COUNTS=Dict{String,Int}()
 @eval ext _hdm_study_count!(key)=(_HDM_STUDY_COUNTS[key]=get(_HDM_STUDY_COUNTS,key,0)+1;nothing)
 source=read(joinpath(dirname(pathof(H)),"..","ext","HighDimensionalModelsProximalAlgorithmsExt.jl"),String)
 expressions=String[]
 for prefix in ("(f::RegressionLoss)(B)","(g::NuclearPenalty)(B)","PA.prox!(Y, g::NuclearPenalty")
  line=only(filter(l->startswith(l,prefix),split(source,'\n')))
  lhs,rhs=split(line," = ";limit=2)
  key=startswith(prefix,"(f") ? "loss_value_calls" : startswith(prefix,"(g") ? "penalty_value_calls" : "optimizer_prox_calls"
  push!(expressions,lhs*" = (_hdm_study_count!(\"$key\"); "*rhs*")")
 end
 if occursin("function PA.value_and_gradient",source)
  first=findfirst("function PA.value_and_gradient",source).start
  last=findfirst("# ProximalCore",source).start-1
  gradient=source[first:last]
  gradient=replace(gradient,"function PA.value_and_gradient(f::RegressionLoss, B)"=>"function PA.value_and_gradient(f::RegressionLoss, B)\n  _hdm_study_count!(\"value_gradient_calls\")")
  gradient=replace(gradient,"    value = f(B)"=>"    _hdm_study_count!(\"cancellation_fallback_calls\")\n    value = f(B)")
 else
  line=only(filter(l->startswith(l,"PA.value_and_gradient(f::RegressionLoss"),split(source,'\n')))
  lhs,rhs=split(line," = ";limit=2)
  gradient=lhs*" = (_hdm_study_count!(\"value_gradient_calls\"); "*rhs*")"
 end
 push!(expressions,gradient)
 first=findfirst("function nuclear_panoc",source).start
 last=findlast("\nend",source).start-1
 native=source[first:last]
 native=replace(native,"  function certified(B)"=>"  function certified(B)\n    _hdm_study_count!(\"certificate_calls\")")
 native=replace(native,"    count[] += 1"=>"    count[] += 1\n    _hdm_study_count!(\"outer_callback_calls\")\n    count[]>1 && state.tau<1 && _hdm_study_count!(\"iterations_with_backtracking\")")
 push!(expressions,native)
 replay=join(expressions,"\n\n")
 Base.include_string(ext,replay,"exact-native-method-replay-after-all-timers.jl")
 bytes2hex(sha256(source)),bytes2hex(sha256(replay))
end
