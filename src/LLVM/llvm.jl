#=2013 LLVM Developers’ Meeting: “Julia: An LLVM-based approach to scientific computing”
by Mr. Keno Fischer - Harvard College/MIT CSAIL "Computer Scientist"

slides 
https://llvm.org/devmtg/2013-11/slides/Fischer-Julia.html#/7

Video: https://www.youtube.com/watch?v=UsdQcbpVWdM&ab_channel=LLVM
2013 - 
Benchmarks = readtable("http")

=#

#=Agenda

- Part1: Why LLVM 
- Part2:
    2.1. introduction
    2.2 High-level Overview of how we go from JuliaCode to LLVM 
    2.3 How we use LLVM as a Part of Julia 
=#

#=Is It any fast?

    When introducing a scientific COmputing Language, 
    the first Question is:
        Q. Is it any fast? 
       #benchmark with a Log-Scale plot 
       deleted #REASON INVALID URI  

    -short answer: Yes, we Are , however, being fast Is Not What We All About ! 

    -being fast:
    1. Emerges from the way we DO Things 

    2. Something you need: for Scientific Computing 
    But, if we All wanted to do is (to) Be Fast, 
    We wouldn't need a new Programming Language 

    3. It Is Just a Side Effect   that we will pass , but: 
        "It's not the main thing, that we're about "

    with that, let's get into a Brief Overview of what Julia is About 

     =#


# the equivalent of Hello world 
username = run(`git config --get user.name`)



#=Dynamic Multiple Dispatch Everywhere 

    (one of the) the main Paradigms in Julia is 
    this Idea "Multiple Dispatch":

    Definition: every function depends on ALL the Types
    of the Arguments you pass in 

##functions 

1.Convert
 - probably (was) the most overloaded function , now it's got only 10 methods
 - it has in the base system, just withouot loading

 2. the Plus +

 -Highly Polymorphic  with 196 methods 

-for any argument (to a function):
    write the call & compiler will tell which method is selected 


    if we want to convert 1 as int64 to Float64
        it would choose the (proper) convert method


    the plus + (between float & int )

    - is a more generic function 

    - that Dispatches based on the number Type 
        (which is a common SuperType of both of them  )
        that's called promotion()   - later
    =#

convert  # 10 method generic function  

+ #196 method generic function (still the largest)

@which convert(Float64, 1) #no unique method found #checking uniqueness of method (if method has n= 1 passed with it )

@which 1.0 + 2 # can add 2 numbers of different types i.e +( float, Int64)
#=+(x::Number, y::Number) via promotion

=#


#= Basic Dispatch

the first function, its first argument: Anyy, unconstrained (can be anything ) -accepts different variables & types 

the second function has with both Arguments of Abstract type Number 
            we have `Type Heirarchy` 
      we do `Type Annotations` by using  ::Any 
        A return type is `implicit` (when no type is specified) 
        but, if want to have more specific, use  Annotations ::

       In Julia,  we don't have (Structural) Inheritance,an idea of sth we may do (in the future) -haven't missed it (so far that much)

       in Julia, we  define a Type Heirarchy, that  we have 
       1.1. Abstract Type :  like Number & Integers (inherits from Number), which everything inherits from it (i.e. floating points Floats, Ints (Int64,32,..),Rationals
       1.2. Concrete Type: which are leaf types (i.e implementable), &  have Fields (for specidic application)

=#
#--- Basic Dispatch

f(a::Any, b) = "fallback" # a generic fallback 
f(a::Number, b::Number) = "a and b are both numbers" # both of Type Number 
f(a::Number, b) = "a is a number" # second argument isa ::Any 
f(a, b::Number) = "b is a number"
f(a::Integer, b::Integer) = "a and b are both integers" #we could be more specific (constrained)

f(1, 5.2) # "a and b are both numbers"
f(1, "bar") #= "a is a number" =#
f(1, 2) #= "a and b are both integers" =#

f("foo", [1, 2]) # : "fallback" -- it compares all functions, none has array explicitly specified, hence  it picks  the `fallback``  (because its 2nd  argument is dynamic ) # Infer 1st: f(a::Any, b) = "fallback"


methods(f) #returns all different methods (& its different types function has ) #convenient
#=julia> methods(f)
# 5 methods for generic function "f":
[1] f(a::Integer, b::Integer) in Main at c:\Users\adamus\.git\DeepLearner\DeepLearner\Tutorial\LLVM\llvm.jl:32
[2] f(a::Number, b::Number) in Main at c:\Users\adamus\.git\DeepLearner\DeepLearner\Tutorial\LLVM\llvm.jl:29
[3] f(a, b::Number) in Main at c:\Users\adamus\.git\DeepLearner\DeepLearner\Tutorial\LLVM\llvm.jl:31[4] f(a::Number, b) in Main at c:\Users\adamus\.git\DeepLearner\DeepLearner\Tutorial\LLVM\llvm.jl:30[5] f(a, b) in Main at c:\Users\adamus\.git\DeepLearner\DeepLearner\Tutorial\LLVM\llvm.jl:28  
=#
#=--- Diagonal Dispatch/ Parametric Dispatch =#

f{T<:Number}(a::T, b::T) = "a and b are both $(T)s"
f{T<:Number}(a::Vector{T}, b::Vector{T}) = "a and b are both vectors of $(T)s"

#=Crossreference  Dispatch on tuples of types https://github.com/JuliaLang/julia/issues/10947
#10947
(Int,) is not a type at all; it's a tuple that contains a type. So you can't use <: on it.

Tuples now "infer" their parameters from constructor arguments just like other parametric types do:
=#
typeof(Int => String) #Pair{DataType,DataType}
typeof((Int, String)) # Tuple{DataType,DataType}
[Int] #1-element Array{DataType,1}: #Int64 # Core.Int64 

#=
The difference is that tuples don't provide a way to request parameter types. If you want, you can construct Pair{Any,Any}(1,2), but you can't do the same thing with tuples. Tuple "parameters" are always inferred.

This is normally what you'd expect, but the corner case is that it's impossible to construct a Tuple{Type{Int}}.

To do this dispatch, I think you'll have to pass tuple types --- dispatch on Type{Tuple{T,T}}. Unfortunately the input syntax Tuple{A,B} is much less pleasant than (A,B). Another example in favor of using {A,B} for tuple types.

The difference is that tuples don't provide a way to request parameter types. If you want, you can construct Pair{Any,Any}(1,2), but you can't do the same thing with tuples. Tuple "parameters" are always inferred.

This is normally what you'd expect, but the corner case is that it's impossible to construct a Tuple{Type{Int}}.

To do this dispatch, I think you'll have to pass tuple types --- dispatch on Type{Tuple{T,T}}. Unfortunately the input syntax Tuple{A,B} is much less pleasant than (A,B) 
Another example in favor of using {A,B} for tuple types.
=#

isa(Int, Any)
#true
isa(Int, Type{Int})
#true
isa(Int, DataType) #true
isa(("",), Tuple{Any}) #true
isa(("",), Tuple{String}) #true
isa(("",), Tuple{ASCIIString}) #true #2022: Error UndefVarErro (is this normal?)
isa((Int,), Tuple{Any}) #true
isa((Int,), Tuple{Type{Int}}) # false
isa((Int,), Tuple{DataType}) #true
#Edit: Actually, upon further reflection, I understand now. typeof((Int,)) is a Tuple{DataType}. And this is behaving like dispatch:

f(::Tuple{Type{Int}}) = 1 #f (generic function with 1 method) #works 

f((Int,)) #ERROR: MethodError: `f` has no method matching f(::Tuple{DataType}) # Closest candidates are: f(::Tuple{Type{Int64}}) #me : so the ....
#ERROR: MethodError: `f` has no method matching f(::Tuple{DataType}) Closest candidates are: f(::Tuple{Type{Int64}})

#= Warning f(::Tuple{Type{Int64}}) #f(::Tuple{DataType}) #OPEN! #BUGGY ERROR Detected (for more review: src\UI\Gtk\issues\open\isa_issue.jl for more examples of buggy code)
#=Remark:in 2013, pointing to a hidden bug, with a one word f((Int,)) -that breaks Julia itself- was later discovered by chance in 2015; till today the issue is open (because it is as deep as LLVM)
the presenter is the least, but the fellow 'scientists' that he resemble are, hereby, spooky=#
