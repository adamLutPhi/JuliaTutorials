
#--- show julia builds code

"""that can give you insight of types (of your code) 

1.text Files read as a string 
2.-julia lets Files gets parsed into Julia `Expr`
`Expr`: a structured representation of the code (you wrote)

3.it organizes structures (for you)- you can manipulate, 
so that you don't do that (as a form of Raw Strings)

e.g. 
Matching the paranthesises

- Expressions get lowered (@code_lowered) to this lowered representation llvm
e.g. you're v. familiar in seeing this message
the `Info: pre-compiling...` message, when you update a package (or install a new one)
Tip: this (llvm) is mosly what gets saved to a file when that happens 


juila is parsing text files into Expressions
2. then lowering them 
3. then julia saves the lowered code to a file 
(on your harddrive)

it can save that work, next time you run julia 

the package is asked for additional information (to be stored )
    
- There is more information into those pre-compile files - only most of what's happening (code also turned to be native )
#--------

when you call `f(args...)` either 

-the native code gets looked up in the in-memory torage & then runs (typo @ 22:42)
OR:

1. f gets type-inferred for the argument types 
2. th type-inferred code gets optimized by Julia (this gets stored in memory)
3. the result gets compiled by LLVM (this also gets stored in memory)
4. Julia runs the compiled code 

when you use a package, it loads that past file - that has lowered code stored in it 
whenever you make a function call, either
    if already compiled, julia has it in memory 
        it will look it up , & run it 
        or 
    if never called it before,  then 
    Julia will Perform type inference for specific types of arguments that 
    you're passing in , 
    that type-inferred code gets optimized (optimize=true stage)
    result gets compiled by an LLVM
    finally, it will run the compile code 
#------Dispatch runtime vs compile-time 

Hoping you'd appreciate the importance of types to Julia's performance


Remember the addTwo 
this leaves a question open 

what is this + here? 

+ is a function call , generic , 200 methods (more if we loaded additional packages)
(as packages can extend functions)

How does julia know which (of the methods)is the right one? 

It chooses the `Most specific method` for the the types of an argument 

types of the argument(here ): x[1] , x[2]
could've been written as:
+(x[1] , x[2]) #then, it'd be more appearant this is a function call , by plus 
which is the applicable one here- 
we can see the decision process using @code_typed (it hates the UndefVars )
give all arguments a value for  addTwo( x[1] , x[2])
    @code_typed: answers which of methods gets called 
"""
addTwo(x) = x[1] + x[2]
"""adding 2 numbers"""
#we'll review this code 
#@code_typed optimize = false addTwo(1, 2.0) #Any[] #no problem

#again rewriting in another way:
addTwo(x) = +(x[1], x[2])
"""
CodeInfo(
1 ─ %1 = Base.getindex(x, 1)::Int64
│   %2 = Base.getindex(x, 2)::Float64
│   %3 = (%1 + %2)::Float64
└──      return %3
) => Float64
"""
@code_typed optimize = false addTwo((1, 2.0)) #Any[]
#=
since it's a typed code, it can figure out which method is the one 
(applicable for those 2 types):
while compiling this code, 
    figure out which method is the one applicable 
for those 2 types 
-the point: it didn't have to run the code (to figure it out )

the Magic: type-infernce : when i run this code, what types will %1 & %2 be ?

If know with certainty, then I can figure out which method will 
Be applicable to them  (at the time I'm building the code ) - not at the time I'm running 

here's julia main Performance bit of  magic: 
takeaway: 
instead of running it with optimize=false  , can see having figured out which method is gonna be called, in optimize=true 
=#
@code_typed optimize = true addTwo((1, 2.0)) #optimize=true performing inlining: inline_expansion

"""
 this ability to 'pear-in the future' 
 & look-up the functions ahead of time 
 is again a key component 

 this is called 
 "Compile-time Dispatch"

 meaning choosing hich method 
 will get executed happens when the calling function is being compiled 

alternative is called runtime Dispatch
we're gonna defeat the ability of julia's 


takeaway:type-inference engine (will) work
by creating a Vector that contains "No Information about the types (of objects) stored in it  "

others like: vector int , Float64, nut this line will gonna create a vector of Any 
"""

@code_typed optimize = true addTwo(Any[1, 2.0]) ## call irt on a Vector(Any) 

#=
Literally any kind of Object, String , a Dictionary,an image, anything you want  
Might be stored in the slots of this Vector 
takeaway: Julia Cannot Predict the types of the Object that will be gen_call_with_extracted_types_and_kwargs  under any circumstances, 

therefore eventhough we turned on optimize = true , you'd hope julia will do all types of inlining 
-heere thre's nothing to inline - it has no clue what method 
+ will be applicable in this circumstance

the first object of type Any - julia has no Idea what type it is 
2nd tye: inferred to be a type Any , therefore + is also any  
why: if it doesn't know the inputs gonna be , it can't figure out what the 
output is gonna be 

julia knows very very little here  
 & it's not made any sort of progress on optimizing your code 
 for  this particular caes 

this ould have then to be executed by a runtime dispach 
=#

"""
1.It knows this arrayref is gonna be called (that's progress of perform)
2.It Doesn't know what the result will be 
that reason it knows 

vector of any isn't the type 
vector of any isn't the type (that julia understands)
that part is a concrete type can figureout 
what's gonna be called here 

but it doesn't know what the result will be  

takeaway: the key decision has to be made, once called the + operator 

(it depends on the arguments)

Only piece of information, julia has to go on  - It doesn't get very far 

means: this method will be looked up & called when the function is running 
-this is called run-time dispach:
schemeatuc of Runtime vs compile-time Dispatch

1. preparing the Arguments
2. Deciding which specifc  compuled blob to use.like looking up someone's 
phonenumber in the phone book 

julia literally scans through the method tables - living in a particular memory location 
machine: 
1.calling funcition invloves: 
1.Arguments: getting them ready to be passed in
& then deciding which specific compile blob to call 

like clicking-up somebody's phonenumber (in a phonebook) [me:Database Analogy]
-which page, column 
-it may take time flipping pages** then look-up the phonenumber 


* that is already been debated: for a phonebook, it's apriori ordered by the first letter
this includes a huge proning of the Search Space 


#--- Schematic of a compiletime call in psuedo-julia 

push!(execution_stack, args)
@goto compiled_blob_52383

the blob will retrieve the argument values by popping the execution execution_stack


-get the arguments ready to go 
-transfer execution (to the blob of compile code) [super-fast]

schematic of a runtime in psuedo-Julia:
1.first: figure out which blob you're gonna get -for the types of arguments 
(this happens during the running of a function)
2.look-up process: paging through the method-tables 
trying to figure-out which is gonna be applicable to this 
particular set of arguments is the slow process 

# scan the method tables and their lists of compiked blobs for a match 
# if the right blob hasn't been compiled yet, compile it now 

"""
blob = get_blob_forargtypes(f, typesof(args)) #Error types of not defined

# the rest looks the same as a compiletime call: 
push!(execution_stack, args)
goto(blob)

#---comparing performance of runtime vs compile-time dispatch 
#=built-in @time macro 
    handy for something, but it's a very fast function=#

using BenchmarkTools

@btime addTwo(z) setup = (z = rand(2))
""

"""
 0.001 ns (0 allocations: 0 bytes)
3.0
it's solved with a nano-second with the cpu codeblock (me:minimum)
"""


@btime addTwo((1, 2.0)) setup = (x = rand(1:5); y = rand())


"
we wanna generate  a random integer between 1 & 5 for x 


 because these are not predictable*

 2.300 ns (0 allocations: 0 bytes) & a random floating number for y #minimum



2.0372456559896017
----------*footnote:
predictability depends on: 1. rngType (subsequently seed choice #TODO:check persistence of a seed) 2.rngSampler: the 'intended' expected shape 

"""
@btime addTwo((x, y)) setup = (x = rand(1:5); y = rand())

"""
  2.300 ns (0 allocations: 0 bytes)
2.002507990680424
pretty much sure that is has been solved
"""
@btime addTwo((x, y)) setup = (x = rand(1:5); y = rand(); z = Any[x, y])

"""
this comes from fact julia has to do method look-up  , at run-time -rather than compile time 

takeaway: make these benchmarks a handy-dandy tool for 
Ballpark costs of runtime dispach (me: gives a rough sketch of performance)

1. single argument: 15-35ns [hardware specific ]
2. two arguments: ~100ns 

cost of a single run-time dispatch

whether it's trivial or catastrophic, it totally depends on what you're Doing

1.Trivial:
If this is 1 operation , it will happen once 
Over the course of your computation 
completely utterly irrelevant 

2. complicated
if it's in an inner-loop (of some computation)
that you're running a billion times 
[billion x 1 op 15-35ns]

Many operations start out taking a large fraction a minute 
After a few optimizations, it's a tiny fraction of a second 

Eliminating run-time dispatch (for newbs) is a single most common thing they could do 

Runtime dispatch isslow 

#--- Union-splitting (new)

an intermediate case 
-julia Can determine that there are only a few possible arguemnt types 

instead of branching your code (checking each type & element it is )

here we know with certainty that 
they are (a choice) [from a UnionSplitting] either:
1.Float32
2.Float64


what we 
  4.600 ns (0 allocations: 0 bytes)
0.7754799894275148

Julia checks what type of arguemnt is 
Then check 1 type of blob 

If it's the other type then it will use the other compile blob 

Warning: there are no blobs looked up at run-time

You just don't know which ones are applicable 

It can check that at runtime 
But especially if these types are  concrete types 
There is only one such type  that fits that description 


"""

@btime addTwo(z) setup = (x = rand(Float32); y = rand(); z = Union{Float32,Float64}[x, y])


argtypes = typeof(args)
push!(execution_stack, args)
if argtypes === Tuple{Float32}
    @goto compiled_blob_52383

else #he only other option is Tuple{Float64}
    @goto compiled_blob_52951 #demo #donotRun

#=
Note: the absence of the need to call `get_blob_got_argtypes` 
Uion-splitting generalizes compiletime dispatch 

now debug code make it working, let's start profiling


=#
