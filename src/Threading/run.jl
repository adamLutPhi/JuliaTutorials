using Threads
using BenchmarkTools
#--- with @threads macro 

arbitraryfun() = 10 * 4 + 1
function nonthreading1(n = 10)
    a = rand(n)
    for i = 1:n
        a[i] = arbitraryfun()
    end
end

"""
- The iteration space is split among the threads, after which each thread writes its thread ID to its assigned locations:
- Note that Threads.@threads does not have an optional reduction parameter like @distributed.

"""
function threading1(m = 4, n = 10)
    a = rand(n)

    @threads for i = 1:m
        a[i] = Threads.threadid()
    end

end

#--- 

#threads = 1
@btime nonthreading1()  #    65.682 ns (1 allocation: 160 bytes)
@time nonthreading1() #  0.000006 seconds (1 allocation: 160 bytes)
@benchmark nonthreading1() # memory estimate:  160 bytes, allocs: 1, minimum time: 62.729 ns (0.00% GC),median time: 70.672 ns (0.00% GC), mean: 96.366 ns (2.57% GC), maximum: 1.227 μs (84.42% GC)

#threads = 4
@time threading1()
# 0.033583 seconds ([17.04 k allocations]: [1004.740 KiB], 99.66% compilation time) 0.000047 seconds (8 allocations: 784 bytes), 2.278 μs (7 allocations: 768 bytes), 0.000030 seconds (8 allocations: 784 bytes)
@btime threading1() #    2.167 μs (7 allocations: 768 bytes) , 2.289 μs (7 allocations: 768 bytes)
@benchmark threading1() ## memory: 768 bytes, allocs: 7, minimum time: 2.256 μs (0.00% GC),median:2.689 μs (0.00% GC),mean:3.473 μs (1.63% GC),maximum: 578.122 μs (97.81% GC)

#threads =  10 
@btime threading1(10, 10) # 75.410 ns (1 allocation: 160 bytes)
@time threading1(10) #  0.000004 seconds (1 allocation: 160 bytes)
@benchmark nonthreading1() # memory estimate:  160 bytes, allocs: 1, minimum:75.410 ns (0.00% GC), median: 82.329 ns (0.00% GC),mean: 106.361 ns (2.37% GC),maximum: 1.776 μs (0.00% GC)
# note for 10 threads, it sounds pretty overwhelmed  

#--- threading2 with @inbounds  factor 

function threading2(n = 10, m = 4)
    a = rand(n)
    Threads.@threads for i = 1:m
        @inbounds a[i] = Threads.threadid()
    end
end


@btime threading2() # 2.256 μs (7allocations:768bytes)
@time threading2() # 0.036257 seconds (17.18 k allocations: 1013.975 KiB, 99.73% compilation time)
@benchmark threading2() #memory:768 bytes, allocs:7, minimum time:2.256 μs (0.00 % GC), median time:2.667 μs (0.00 % GC), mean time:3.297 μs (1.68 % GC), maximum time:[560.533 μs] (99.03 % GC)


function nonthreading2(n = 10)
    a = rand(n)
    for i = 1:n
        @inbounds a[i] = arbitraryfun()
    end
end


@btime nonthreading2() # 63.988 ns (1 allocation: 160 bytes)
@time nonthreading2() # 0.000003 seconds (1 allocation: 160 bytes)
@benchmark nonthreading2() #   memory estimate:  160 bytes, allocs:1, minimum: 61.554 ns (0.00% GC), median: 69.018 ns (0.00% GC), mean: 90.957 ns (2.74% GC), maximum: 1.789 μs (0.00% GC)


@inbounds function nonthreading3(n = 10)
    a = rand(n)
    for i = 1:n
        @inbounds a[i] = arbitraryfun()
    end
end


@btime nonthreading3() #  63.988 ns (1 allocation: 160 bytes)
@time nonthreading3() # 0.000006 seconds (1 allocation: 160 bytes)
@benchmark nonthreading3() # minimum: 64.090 ns (0.00% GC), median: 71.211 ns (0.00% GC), mean: 104.758 ns (2.84% GC), maximum:     2.561 μs (95.54% GC)

#=
adding @inbounds to a function name slightly lags processing time  #negligable
=#


@inbounds function nonthreading4(n = 10)
    a = rand(n)
    @inbounds for i = 1:n
        @inbounds a[i] = arbitraryfun()
    end
end

@btime nonthreading4() # 64.249 ns (1 allocation: 160 bytes)
@time nonthreading4() # 0.000008 seconds (1 allocation: 160 bytes)
@benchmark nonthreading4() # memory estimate:   memory estimate: 160 bytes, allocs: 1,minimum time:     61.185 ns (0.00% GC), median: 68.539 ns (0.00% GC), mean: 92.921 ns (3.10% GC),maximum:     1.504 μs (86.53% GC)