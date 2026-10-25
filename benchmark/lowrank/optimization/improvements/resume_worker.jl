# Resume only flushed timing rows and completed profile files after interruption.
# The frozen numerical kernels and measurement function are unchanged.
ENV["CD81_IMPROVEMENTS_LIBRARY"]="1"
include(joinpath(@__DIR__,"updated_worker.jl"))
function resume_main()
  input,out,variant = ARGS[1:3];mkpath(out);d=problem(input)
  memory = length(ARGS)>=4 ? ARGS[4] : ""
  solvers = variant == "before" ? BEFORE : AFTER
  !isempty(memory) && (solvers=[memory])
  csvpath=joinpath(out,"julia.csv")
  done=Set{Tuple{String,String,Float64,Int}}()
  hasheader=isfile(csvpath) && filesize(csvpath)>0
  if hasheader
    lines=readlines(csvpath)
    valid=String[first(lines)]
    for line in lines[2:end]
      parts=split(line,',')
      length(parts)==length(FIELDS) || continue
      solver,mode,frac,rd=parts[6],parts[7],parse(Float64,parts[8]),parse(Int,parts[9])
      if mode=="fit" && rd==1
        coefficient=joinpath(out,"$solver-$frac.bin")
        isfile(coefficient) && filesize(coefficient)==8d.p*d.q || continue
        if updated_family(solver)=="sparse"
          all(isfile(joinpath(out,"$solver-$frac.$tag")) for tag in ("A","V")) || continue
        end
      end
      push!(done,(solver,mode,frac,rd))
      push!(valid,line)
    end
    open(f->foreach(line->println(f,line),valid),csvpath,"w")
  end
  warm=problem(joinpath(dirname(input),"small-1"))
  warmwide=merge(warm,(;n=30,X=warm.X[1:30,:],Y=warm.Y[1:30,:],layer=warm.layer[1:30,:]))
  for s in solvers, w in (warm,warmwide)
    execute(s,w,.1)
    if updated_family(s)=="nuclear"
      execute(s,w,.1;mode="kernel",cached=cache(w))
      execute(s,w,.1;mode="path")
    end
  end
  RSS_BASELINE[]=Sys.maxrss()/1024
  println("Julia ",VERSION," BLAS ",BLAS.get_config()," threads=",BLAS.get_num_threads())
  open(csvpath,"a") do io
    !hasheader && println(io,join(FIELDS,','))
    if !isempty(memory)
      measure!(io,d,out,memory,.1,1,variant;mode="memory")
      return
    end
    for rd in 1:3, s in circshift(solvers,mod(d.rep+rd,length(solvers)))
      for frac in (updated_family(s)=="rank" ? (0.0,) : (.5,.1,.02))
        (s,"fit",frac,rd) in done || measure!(io,d,out,s,frac,rd,variant)
      end
    end
    nuc=filter(s->updated_family(s)=="nuclear",solvers)
    for s in nuc, mode in (variant=="before" ? ("path",) : ("kernel","path")),rd in 1:3
      (s,mode,mode=="path" ? .02 : .1,rd) in done || measure!(io,d,out,s,.1,rd,variant;mode)
    end
    rp=variant=="before" ? "before_rankpath" : "cd_rankpath"
    for rd in 1:3;(rp,"fit",0.0,rd) in done || measure!(io,d,out,rp,0.0,rd,variant);end
  end
  if d.case in ("medium","responses","wide")
    specs=[(s,"fit") for s in solvers]
    push!(specs,(variant=="before" ? "before_rankpath" : "cd_rankpath","fit"))
    append!(specs,variant=="before" ? [("before_nuclear","path")] : [("cd_nuclear","path"),("cd_panoc","path")])
    for (s,mode) in specs
      suffix=mode=="path" ? "-path" : ""
      profilepath=joinpath(out,"profile-$s$suffix.txt")
      isfile(profilepath) && filesize(profilepath)>0 && continue
      Profile.clear();began=time();calls=0
      @profile while time()-began < .25 || calls==0
        execute(s,d,.1;mode);calls+=1
      end
      suffix=mode=="path" ? "-path" : ""
      open(joinpath(out,"profile-$s$suffix.txt"),"w") do f
        println(f,"case=",d.case," rep=",d.rep," calls=",calls," samples=",length(Profile.fetch()))
        Profile.print(f;format=:flat,sortedby=:count,mincount=2)
      end
    end
  end
end
resume_main()
