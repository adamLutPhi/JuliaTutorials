"""

A fast language can be beat 
by a slow one `if` the code in the slow language solves the problem
more efficiently 
 (if slower is uses a  smarter, better Algorithm )

takeaway: Implementation:
1. Language
2. Solution Strategy


You can also get the opposite 

In a slow language , the converse can also be true 

A poor algorithm might run faster than a good algorithm 
(due to language limitations): #unlike Julia (doesn't have overhead, has optimization)
Tim Holy: the actual `impractice performance` you get from a language that doesn't do the Optimziations(julia does ) depends upon 
what's Built-in for you 
basically relies on people having written  `the things` that your particular Implementation requires 
in a way that allows it to run quickly
invariably, you'll find certain concepts are efficient 
other construcs arent' efficient
[some examples would change your impression on how the best way to solve a problem is]


Certain constructs are implemented efficiently 
(Certain other constructs are not )


"""

"""Big-O Notation @22:34

you may notice there's a little work outside the Innermost loop 
Point of Big-O 
freedom to ignore terms that are less imprtant 
me: I have to digress, it's a measure of complexity , as implied by demonstrating an inner-loop i.e. raising the already complex complexity 
-Tim: "you might object", say  you're ignoring certain things  !
"""

for i = i:m 
    s=0              # happens `m` times  - little work before time 
    for j = 1:n 
        s += M[i,j] * v[j] # happens `m*n` times 
        
    end
    out[i] = s          # happens `m` times  # little work after time 
end 
#=more accurately, the running time is `a*m + b*m*n`, where a &b are constants # the actual running time 
(since inner-loop runs m*n times )

Takeaway: Pay attention to the things that grow most quickly
so O(mn)
this algorithm is big-O -            a lisence to ignore small ones, yet focus on big ones that matter 
=#

"""Key Rules about memory @41:38

1. Computers have multiple layers of caches 
2. the fastest caches have only ~1/10^6 of storage of main memory 
3. Main memory gets transferred to cache in small chuncks of adjacent memory (cache line)
4. Performace is Better If you Exploit what you have in cache before flushing it & swapping in new Memory
5. CPUs try to Predict what Chunck will be Needed next & start fetching it before 

Tip: `LoopVectorization` will Reorder Your Loops & Make Other changes that help not Slow Code become "fast"
"""

"""Know the Costs & Benefits of Threading 

A `Thread.@threads` has an overhead of a few microseconds. 
equivalnet to 1000s of computations 
(Reverse threding for sizable jobs)


- In cases where threads can work independently. it's sometimes possible to get an `~n` fold speedup from `n` threads 


Communicating between threads can be a source of bugs &/or slowdowns 

sum values of vector v 
threaded loop (without it would look the same as normal loops) - enable us to Distribute the work 
"""


function threadedsum(v)
    s = 0
    Threads.@threads for i in eachindex(v)
        s = s + v[i] #readout old value, then increments with next item on the list, store result back in s again
    end
    return s
end

#Define v
@time v = rand(1:5, 10^4) #0.000105 seconds (2allocations:78.203KiB)

@time threadedsum(v) # 0.092380 seconds (93.25 k allocations: 5.186 MiB, 99.55% compilation time)
# 30144
@time sum(v) # 0.017116 seconds (262 allocations: 13.641 KiB, 96.04% compilation time)

"""Race-condition 

one of common things to happen (me:is it because not locking the common working memory? ) - possible 
multiple working threads 

3 working fast 
1 of them gets interrupted 
it gets that core, to do sth else 
then it comes back,  picks off execution, where it left off 

read out a v. old value of s 
we don't know its' refreshed 
then overwrite the value you had before 
this causes it to drop many terms of v 

sum (subset (value of v) ) 

Bugs arise from one thread being overwritted by another thread 

1 way : 
prevent 1 thread from accessing s 

caled blocks new texes, atomic operation (happening in time )
in this case 

problem blocking (& its generalizations)
while one
one thread holding lock, other (simply) has to wait - until the first one is done 
(that delays the thread )
Unfavorable circumstances that take way way longer than a simple implentation 
Solution:
by giving each threads to Maintin their counter 


"""
function threadedsum2(v)
    spartial = zeros(Int, Threads.nthreads()) # s = 0 #allocates a n(counter) = to n(threads) 
    Threads.@threads for i in eachindex(v)
        #s = s + v[i] #readout old value, then increments with next item on the list, store result back in s again
        spartial[Threads.threadid()] += v[i]  # each counter implements just its own counter
    end
    return sum(spartial) #return sum of 4 values Accumulated by each thread  
end


@time threadedsum2(v) # 0.000087 seconds (9allocations:736bytes)

sum(v)

"""Other opportunities in parallelism 

1. GPU Computing: julia can generate native code for GPUs (see JuliaGPU)
2. Distributed computing clusters 
- each server is working on the portion of the same problem 

cluseter computing is supported by julia support library (& other packages in julia ecosystem)

-Caution: in slow languages, sometimes people reach almost unthinkingly for parallelism 
So if you're coming from a slow language, you may essentially reach parallelism almost inflexible 
You get most out of it optimizing single mode 
You'll iron out any performance gotchas (wasts lots of time ) -regardless if you parallize it 
-Julia's advantage , you don't need parallelism to get the speed you need 

Then, if you have done all that , really taken a serious look at your code 
Good algorithm, 
Implemented it well, using Idiomatic Julia ( & if it's still not passing up)
Then  that's the time to reach for parallelism
It's a well justified choice (in that circumstance )
But think it's worth emphasizing how important it is to 

1.First, try to optimize simpler implementation -that's all you need- for many practical problems 
2.When you Parallelize it, you'll get a Better Result - for having done so 


good notes:

it's important to use julia well, because you can handicap your code (tell me about it)
and there's consequences of your performance quite severe if you do so 

the goodside: there aren't many things you have to know in order to write good idiomatic julia code 
you can then use julia pragmatic experience -testing differnet thing out-to quickly develop intuition about 
what makes a good algorithm
it's a more reliable ally to your inituition
it really helps you grow as a programmer 



Next steps :

1. Algorithms & data structures: Hash tables, Heaps, Priority Queues, et al.
2. extending knowledge of idiomatic Julia: read package code & Julia's `Base` Module
3. learning more about threading & High-performance computing 

This course is a taste of lots of things that Prof. Tim Holy found useful 
There's a lot out there - many fish in the sea
Tastes offered is just a basis 
Exploration is a v. worthwhile experience 
Active maintainer: continously spending time  as an 
Odds are: v. good! you can get accustumed tuturing from more experienced julia development 

