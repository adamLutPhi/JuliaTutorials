#---profiling 
#=NEW - Profiling
once code is writting, running 
we hat to profile it ens then you

1.Start tuning: the performance of code 

why: once working, figure out where it spends its time
-Timing the execution is very informative 

Takeaway: but as code gets more complicated, 

        "Measuring time become Less & Less Useful"

2.start guessing: which part of my code is  
 eating up all time (optimize:if you can somehow lower that down)

for that: encourage you to use a set of Profilers 


Profiling: what parts of code are Taking part of the time 


(in very rough terms) there are two types of profilers:
# instrumenting profiles
an instrument profiler adds measurement 'instrumentation' to your source code 

say call f(of x) store results in y: y = f(x) #me: calling f(x) a single variable function 


=#
using BenchmarkTools

y = f(x) # this is line

z = g(x, y)

"""
when you run code under instunmenting profiler 

it will actually modify that code (effectively)

& run a code looks like this :
"""

push!(timebuffer, ProfileInfo("somefile.jl", 17, time()))
y = f(x)
push!(timebuffer, ProfileInfo("somefile.jl", 18, time()))

z = g(x, y)

push!(timebuffer, ProfileInfo("somefile.jl", 19, time()))

"""basucally it has some buffer that keeps track of timing data
 Right before it runs(first line), it might basically
 Record some information about 
1 .where it is (in your source code)
2. what the current time is 

then it will make that work 
 """

"""
first it knows what time it is 
queries the system 

(me:simulates time advancesment : by passing i as an argument )
time would have advanced by the function call 
(On) every execution, it gets called 
that's a very nice way of measuring time 

there are a Quite a few profilers, work this way 

however,
Insturment profiles come with hurdles:

1.instrumentation slows your code: having to store data at every single function call #TODO: that implements as well in metatrader 
2.instumentation can block compiler optimizations: the compiled code of the real version may be quire different from that of the instrumented code "minus instrumentation"
**3.recursion is a bit tricky**: if you also instrument `f`, the added instrumentation inside `f` distorts your measurement of the runtime of `f` itself

so, running code with instrumentation is a very different code, than the code that runs without the instrumentation  
you're actually getting a misleading picture of the cirumstances  

anytime you do instrumentation inside a function call 
has a little instrumentation around it -gets complicated to subtract out ]
costs of insturmentation 
world has moved on to a different stle of profiler
"""
i = 17
push!(timebuffer, ProfileInfo("somefile.jl", i, time()))
y = f(x)
push!(timebuffer, ProfileInfo("somefile.jl", i = i + 1, time()))

z = g(x, y)
push!(timebuffer, ProfileInfo("somefile.jl", i = i + 1, time()))


#=Sampling profilers 

a sampling profiler ,periodically, 
1.interrupts your code and 
2. collects program-location data (where you are)

Analogy: 

8 hours a day at work 
1 hour a day at the gym 

real word 
waka-time collecting time intervals  of user code activity plus langauge programmed in 
 shortcoming: does this 100s times over a month
 
 aggregates all responses the
 
 =#
using Makie 
# using GLMakie
location = [rand(1:9) <= 8 ? "work" : "gym" for i = 1:1000]

bins = rand(5:16)
#(values = location, bins = bins, normalization = :none) #TODO: hist equivalent (makie)
#("Counts")


using PyPlot

location = [rand]
location = [rand(1:9) <= 8 ? "work" : "gym" for i = 1:1000]

bins = rand(5:16)
hist(values = location; bins = bins, normalization = :none)
ylabels("Counts")

"""
Disadvantages of sampling profilers
1. you don't collect exhuastive information 
spending 5 minutses at bank will only be captured at all 

2. it's subject to sampling noise (given `n` counts, you have uncertainty
 of approx. `sqrt(n)` counts)

you'll get different number 

Advantages of sampling profiler
you're running unmodified code 

Code has all the real world optimizations (the compiler can do)

"""

#--deno of profiling 


function busywait(t)

        x = 0
        for i = 1:round(Int, 2.1e10 * t)
                x += i % 2
        end
        return x # return `x` to prevent the compiler from noticing that this doesn't do real work & eliminating it 
end

@time busywait(0.8) # 8400000000

function mydays(n)
        x = 0
        for i = 1:n
                x += work()
                x += gym()
        end
        return x
end

@noinline work() = busywait(0.08)
@noinline gym() = busywait(0.01)

mydays(1) # run once to compile it #945000000 

"""NEW
good rule of thumb:
takeaway: clear the buffer of the library before use 

bullet note: the overhead, when the function itself actually is doiing the work

Count  Overhead File                   Line Function
 =====  ======== ====                   ==== ========
  1877         0 @Base\boot.jl           360 eval
  1877         0 @Base\essentials.jl     708 invokelatest(::Any, ::Any, ::...  1877         0 @Base\essentials.jl     708 #invokelatest#2
  1877         0 @Base\essentials.jl     706 invokelatest(::Any, ::Any, ::...  1580      1580 @Base\int.jl             87 +
  1877         0 @Base\loading.jl       1116 include_string(mapexpr::typeo...  1877         0 @Base\logging.jl        603 with_logger
  1877         0 @Base\logging.jl        491 with_logstate(f::Function, lo...  1877         0 @Base\task.jl           411 (::VSCodeServer.var"#53#54")()  
     1         1 ...ile\src\Profile.jl   352 stop_timer()
  1580         0 ...torial\profiler.jl   157 busywait
   190        29 ...torial\profiler.jl   174 gym()
  1686         0 ...torial\profiler.jl   167 mydays(n::Int64)
   190         0 ...torial\profiler.jl   168 mydays(n::Int64)
  1686       267 ...torial\profiler.jl   173 work()

me:  count/overhead ratio : for gym : 

 190/29 =  6.551724137931034
 1686/267 =  6.314606741573034

work()/gym() ratio:
 267/29 = 9.206896551724138 (over 8 , overworking (seemingly, undergyming)  )
 1686/267 = 6.314606741573034 (under 8, underworking (seemingly over gyming))
  
from looping at the sampling profiler data 


"""

using Profile
Profile.clear() # clear old results (not really needed on the first usuage)
@profile mydays(30)

Profile.print(format = :flat) # represents number of snapshots that it's calling


"""another default way prented in t
 Count  Overhead File                   Line Function
 =====  ======== ====                   ==== ========
  1877         0 @Base\boot.jl           360 eval
  1877         0 @Base\essentials.jl     708 invokelatest(::Any, ::Any, ::...  1877         0 @Base\essentials.jl     708 #invokelatest#2
  1877         0 @Base\essentials.jl     706 invokelatest(::Any, ::Any, ::...  1580      1580 @Base\int.jl             87 +
  1877         0 @Base\loading.jl       1116 include_string(mapexpr::typeo...  1877         0 @Base\logging.jl        603 with_logger
  1877         0 @Base\logging.jl        491 with_logstate(f::Function, lo...  1877         0 @Base\task.jl           411 (::VSCodeServer.var"#53#54")()  
     1         1 ...ile\src\Profile.jl   352 stop_timer()
  1580         0 ...torial\profiler.jl   157 busywait
   190        29 ...torial\profiler.jl   174 gym()
  1686         0 ...torial\profiler.jl   167 mydays(n::Int64)
   190         0 ...torial\profiler.jl   168 mydays(n::Int64)
  1686       267 ...torial\profiler.jl   173 work()
  1877         0 ...Server\src\eval.jl   201 (::VSCodeServer.var"#55#59"{V...  1877         0 ...Server\src\eval.jl   124 (::VSCodeServer.var"#56#60"{B...  1877         0 ...Server\src\eval.jl   153 (::VSCodeServer.var"#57#61"{B...  1877         0 ...Server\src\eval.jl   155 (::VSCodeServer.var"#58#62"{B...  1877         0 ...Server\src\eval.jl   211 inlineeval(m::Module, code::S...  1877         0 ...Server\src\eval.jl   209 inlineeval##kw
  1877         0 ...Server\src\eval.jl    34 macro expansion
  1877         0 ...Server\src\repl.jl    36 hideprompt(f::VSCodeServer.va...  1877         0 ...Server\src\repl.jl   185 withpath(f::VSCodeServer.var"...Total snapshots: 1877

Overhead ╎ [+additional indent] Count File:Line; Function
=========================================================
    ╎1877 @Base\task.jl:411; (::VSCodeServer.var"#53#54")()
    ╎ 1877 ...Server\src\eval.jl:34; macro expansion
    ╎  1877 @Base\essentials.jl:706; invokelatest(::Any)
    ╎   1877 @Base\essentials.jl:708; #invokelatest#2
    ╎    1877 ...erver\src\eval.jl:201; (::VSCodeServer.var"#55#59"...       
    ╎     1877 @Base\logging.jl:603; with_logger
    ╎    ╎ 1877 @Base\logging.jl:491; with_logstate(f::Function...
    ╎    ╎  1877 ...ver\src\eval.jl:124; (::VSCodeServer.var"#56#6...        
    ╎    ╎   1877 ...ver\src\repl.jl:36; hideprompt(f::VSCodeServe...
    ╎    ╎    1877 ...ver\src\eval.jl:153; (::VSCodeServer.var"#57#...       
    ╎    ╎     1877 ...er\src\repl.jl:185; withpath(f::VSCodeServe...        
    ╎    ╎    ╎ 1877 ...er\src\eval.jl:155; (::VSCodeServer.var"#5...        
    ╎    ╎    ╎  1877 ...r\src\eval.jl:209; inlineeval##kw
    ╎    ╎    ╎   1877 ...r\src\eval.jl:211; inlineeval(m::Module,...        
    ╎    ╎    ╎    1877 ...essentials.jl:706; invokelatest(::Any, :...
    ╎    ╎    ╎     1877 ...ssentials.jl:708; invokelatest(::Any, ...        
    ╎    ╎    ╎    ╎ 1877 ...\loading.jl:1116; include_string(mape...        
    ╎    ╎    ╎    ╎  1877 @Base\boot.jl:360; eval
   1╎    ╎    ╎    ╎   1    ...\Profile.jl:352; stop_timer()
    ╎    ╎    ╎    ╎   1686 ...profiler.jl:167; mydays(n::Int64)
 267╎    ╎    ╎    ╎    1686 ...profiler.jl:173; work()
    ╎    ╎    ╎    ╎     1419 ...rofiler.jl:157; busywait
1418╎    ╎    ╎    ╎    ╎ 1419 @Base\int.jl:87; +
    ╎    ╎    ╎    ╎   190  ...profiler.jl:168; mydays(n::Int64)
  29╎    ╎    ╎    ╎    190  ...profiler.jl:174; gym()
    ╎    ╎    ╎    ╎     161  ...rofiler.jl:157; busywait
 161╎    ╎    ╎    ╎    ╎ 161  @Base\int.jl:87; +
Total snapshots: 1877

1686/ 1877 = 0.89824

Takeaway: in profiler tree: the indentation indicates who called `whom` 

Information is attributed to the entire call sequence 
(aggregated in al time spent in that particular line of code )

different way of checking, it doesn't take a different amount of time 


TODO: 
Q. what is an overhead, & overcount #Research 

u

"""
Profile.print(format = :tree)

#--visualization of profile data 


using ProfileView # `ProfileView` for real work instead

ProfileView.view()


#= on jupyter only 
using ProfileSVG

ProfileSVG.view()
=#
#--- Performace profiling in action 
#=
Discovering an opportunity for an optimization 
 by a profile 

=#

#Data definition 

A = rand(10_000, 2)
B = rand(2, 8_000)
x = rand(8_000)
function mult(A, B, x)
        C = A * B
        return C * x
end

mult(A, B, x)

@time mult(A, B, x)  # 0.500144 seconds #2.57 M allocations: 748.353 MiB, 4.44% gc time, 77.02% compilation time
using ProfileSVG
#Error:@profview is undefined - which macro is it from?
@profview mult(A, B, x)

"""
(always) know what to look for!
it's revealing this 'gemm' where 
mm is matrix multiply
mv: matrix vectore (multiply)

most of time spent (lost) multiplying those matrices 
-Going back above, 'C' is a matrix multiplication - 
thus, a bottle is easily spottable, now

what if: 
1.we do the matrix multiplication, first 
2. do a Second matrix-vector multiplication 
"""
function mult2(A, B, x)

        y = B * x
        return A * y

end

# the use of \approx or ≈
mult2(A, B, x) ≈ mult(A, B, x) #they are approx. equal 
mult2(A, B, x) == mult(A, B, x) # but they wont ever be equal 
error = mult2(A, B, x) - mult(A, B, x)
#= ME:
critical side-note:
this introduces an unexpected  systematic model error  
=#
"""Tim Holy
here the error is small, 
there's a need for a hight precision  
so, to a very high-degree of precision , these two are equal
(to ecach other )
me: discarding the error, as if it doesn't exist
like blinding turning an eye on it 
it won't help much 


"""

@time mult2(A, B, x) #0.000718 seconds (3allocations:78.297KiB)

#=we've made a dramatic  Improvement on the matrix multipkucatuib 

=#
#--- Using profiling to detect "gotchas"


#=
Recall the addTwo was slopw when passed a `Vector{Any}`
but fast for a `Vector{T}` with concrete `T`` 
=#
#= 0.001 ns (0 allocations: 0 bytes)
(:
:var"#addTwo#5") (generic function with 1 method)=#
@btime addTwo(z) = (x = rand(1:5); y = rand(1:5); z = [x, y])

#=  0.001 ns (0 allocations: 0 bytes)
(::var"#addTwo#6") (generic function with 1 method) =#
@btime addTwo(z) = (x = rand(1:5); y = rand(1:5); z = Any[x, y])

"""Any type makes  running code longer 
Q. what if i did it on accident, what should I do now?

sometimes you'll be surpised about the code you write that breaks type-Inference

#BenchmarkTools have a macro  @bprofile 
Why: to collect profile Data on (code)
me: outputs statistics of  a 10K samples,  1K evaluations 
"""

@bprofile addTwo(z) = (x = rand(1:5); y = rand(1:5); z = Any[x, y])

"""
       
BenchmarkTools.Trial: 10000 samples with 1000 evaluations.
 Range (min … max):  0.001 ns … 1.400 ns  ┊ GC (min … 
max): 0.00% … 0.00%
 Time  (median):     0.001 ns             ┊ GC (median):    0.00%
 Time  (mean ± σ):   0.043 ns ± 0.053 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █                        
    ▄
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▂
  0.001 ns       Histogram: frequency by time        0.1 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.   

 involving strings, 
 comes from times operation on code
 prints in REPL all type info & method instumentationif right click on it 
 it takes you to the code 

 all its copying going on 

 the Hint: the copying (hat's going on)
 we're starting of fwith an empty string 
 2. we're going char by char 
 3. the, you add another Whole character to a string 
 -it's gonna have to copy the old string, with a little extrra space on the end (then the character)
 """

using Profile, ProfileSVG, ProfileView, BenchmarkTools
const charlist = 'a':'z' #Global Variable # define as const # 

function randomstring(n::Integer)
        str = ""
        for i = 1:n
                str *= rand(charlist)
        end
        return str
end

"""preallocate is  the answer
Instead, preallocate some Memory
 
after applying changes, check if it's working, as expected before 
"""


function randomstring2(n::Integer)
        # preallocate some Memory
        chars = Vector{UInt8}(undef, n)
        #preallocate a buffer for this 
        #instead of growing it each time , simply stash the random variable
        for i = 1:n
                chars[i] = rand(charlist) #stash the chars (me:1 by 1)
        end
        #return str 
        return String(chars)
end

function randomstring3(n::Integer)
        # preallocate some Memory
        chars = Vector{UInt8}(undef, n)#local variable #how did he know that, is scope is this function only, isn't that enough to make it local
        #preallocate a buffer for this 
        #instead of growing it each time , simply stash the random variable
        for i = 1:n
                chars[i] = rand(charlist) #stash the chars (me:1 by 1)
        end
        #return str 
        return String(chars)
end

ProfileSVG.view() #initializes ProfileView



randomstring(8)
@time randomstring(10^5);#4.639694 seconds (300.00 k allocations: 4.672 GiB, 16.61% gc time)

randomstring2(8)
@time randomstring2(10^5); #in REPL #0.008138 seconds (99.49 k allocations: 1.709 MiB)

ProfileView.@view()
warntype_last #warntype at last thi clickedd onsa
@time randomstring3(10^5); # global vairable charlist # 0.063154 seconds (155.18 k allocations: 4.939 MiB, 85.53% compilation time)
 #=first page on the ma
 return 
 main.charlist is a global variable
  & not constant 
  since it's not constant - it change the type 
  any moment 
  so, julia can't create an optimized version 
  of this function with Fixed height (length)
 for charlist

one point of View: it could be multi-threaded (we could change its type)
takeaway: we can change type of code of fuliea 
while the code is running 

        so, that would be really bad if it would hit
        hard wired-in Known Type, & that type change 
       that would be really problematic 
if wanna fix  the performance , the way you can do it, declare this list constant 

Takeaway: for any project, determine list of constants, for them to be preallocated each time the project loads 
that will saved from problems & code performace issues  
        =# 

#=
looking at the second method 
what is that line red ?
-it's doing dispatch 
 profile View's other feature 
 it'll show you code Type, or specialized color version
 @code_warn_type (warntype_last()../6) (outputs shows which line) on the last thing you clicked on
=#

#=
julia can be fast, but you need to learn enought to avoid some common gotchas 
1. don't use non-`const` global variables
2. don't use containers like arrays with non-concrete types (unless absolutely necessary)
3. measure & analyze performance with @time, BenchmarkTools,`@Profile`, `@bProfile`,ProfileView
=#
#===end of profiling optimization part 1=====