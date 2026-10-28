function main(args)
  length(args) <= 1 || error("Usage: make-tasks.jl [TASKDIR]")
  dir = isempty(args) ? (@__DIR__) : abspath(only(args))
  mkpath(dir)
  open(joinpath(dir, "tasks.txt"), "w") do io
    for case in 1:8, firstrep in 1:50:500
      println(io, "$case $firstrep $(firstrep+49)")
    end
  end
  write(joinpath(dir, "canary.txt"), "1 1 2\n")
  println("wrote 80 tasks and one canary task to $dir")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
