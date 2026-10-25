using Random, SHA, TOML
root, phase = ARGS
const CASES = (
  ("small",1000,1,false,0,10.,10.),
  ("irregular",10000,1,true,0,1.,1.),
  ("linear100k",100000,1,false,0,1.,1.),
  ("quadratic100k",100000,2,false,0,.1,.1),
  ("tvpath",10000,0,false,50,.1,10.),
  ("linearpath",10000,1,false,50,.1,10.),
  ("quadraticpath",10000,2,false,50,.1,10.),
  ("linear1m",1000000,1,false,0,1.,1.),
  ("quadratic1m",1000000,2,false,0,1.,1.))
reps = phase == "pilot" ? (0:0) : (1:3)
cases = phase == "pilot" ? CASES[1:6] : CASES
mkpath(root)
open(joinpath(root,"cases-$phase.csv"),"w") do io
  println(io,"case,n,order,irregular,path,lo,hi,rep")
  for (ci,c) in enumerate(cases), rep in reps
    name,n,order,irregular,path,lo,hi = c
    folder=joinpath(root,"$name-$rep")
    ispath(folder) && error("refuse to overwrite $folder")
    mkpath(folder)
    seed=98261026+100ci+rep;rng=MersenneTwister(seed)
    t=range(0.,1.;length=n)
    y = sin.(4pi .* t) .+ 0.6 .* (t .> 0.55) .+ 0.2 .* randn(rng,n)
    x = irregular ? cumsum(0.8 .+ 0.4 .* rand(rng,n)) : Float64[]
    for (key,value) in (("y",y),("x",x))
      open(s->write(s,value),joinpath(folder,"$key.bin"),"w")
    end
    meta=Dict("case"=>name,"n"=>n,"order"=>order,"irregular"=>irregular,
      "path"=>path,"lo"=>lo,"hi"=>hi,"rep"=>rep,"seed"=>seed,
      "y_sha256"=>bytes2hex(sha256(read(joinpath(folder,"y.bin")))),
      "x_sha256"=>bytes2hex(sha256(read(joinpath(folder,"x.bin")))))
    open(s->TOML.print(s,meta),joinpath(folder,"meta.toml"),"w")
    println(io,join((name,n,order,irregular,path,lo,hi,rep),','))
  end
end
