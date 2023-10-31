#=
2013 LLVM Developers’ Meeting: “Julia: An LLVM-based approach to scientific computing”
by Mr. Keno Fischer - Harvard College/MIT CSAIL "Computer Scientist"

[@Keno](https://github.com/Keno)

-Slides
https://llvm.org/devmtg/2013-11/slides/Fischer-Julia.html#/7

-Video: https://www.youtube.com/watch?v=UsdQcbpVWdM&ab_channel=LLVM
    Benchmarks - readtable("http") (2013), youtube.com
=#

#=Agenda

- Part1: Why LLVM
- Part2:
    2.1. Introduction
    2.2  High-level Overview of how we go from JuliaCode to LLVM
    2.3  How we use LLVM as a Part of Julia
=#

#= How fast?

    When introducing a Scientific Computing Language,
    the first Question is:
        Q. Is it any fast?
        A.
       #1. Benchmark with a Log-Scale plot
          <deleted> REASON: INVALID URI

    -short answer: Yes, we Are, however, being fast Is Not What We All About!

    ## Being fast:
    Why (3):
    1. It Emerges from the way we Do Things

    2. Required: for Scientific Computing
    -- "if `fast` is the only quirk, we wouldn't need a new Programming Language

    3. A `Side Effect`: that we will pass
    -- "It's not the `main thing` that we're about"

    With that, let's carry on with a Brief Overview of what Julia is About
     =#

# The  `Hello world` equivalent

### 1. Ability to run scripts

# git config
username = run(`git config --get user.name`)

###2.
### 2. Dynamic Multiple Dispatch
    #=
    "Multiple Dispatch":
    (one of the) the Main Paradigms in Julia is

    Definition: every function depends on ALL the Types
    of the Arguments you pass in


    let f be a function ,
     parameter[] is an  of parameters,  passed into `f`

     and T[] be the  an `Array` of type `Type`.
    of the passed parameter
    =#

## Functions

#= we have (3):
### 1
1. `Convert` function:

1.Properties
 1.1  the most Overloaded function Probably was
 1.2  Now, it has (10) methods
 1.3 It is in the `Base` system ( without `loading` prefix)

2. Highly Polymorphic
     with 196 methods (at the time of the Conference date)

3. Callable

- for any argument (to a function):

    2.1 we can (2)

    1. Write the call
    2. Compiler picks which method is selected

    2.2. If we want to convert 1 as int64 to Float64:
        it would choose the (proper) convert method

    2.3.The plus (+): It also has float & int

    2.4 a  `Generic function` oriented:
    -  which is Dispatched [How?]
    -- Based on the number of `Type` variables
        1. see also: `promotion()`
        2.It's a common `SuperType` of both of them `(Type , promotion)`
    =#


println(convert)  # 10 method generic function
# + 196 method generic function (still the largest)

#see also(metaprogramming)[https://docs.julialang.org/en/v1/manual/metaprogramming/]

#Tip: ask questions with `@which`
@which convert(Float64, 1)
#There is no unique method found
#We can check for Whether a method is unique [How?]
# (if a method has n = 1 passed with it )

@which 1.0 + 2 # can add 2 numbers of different types i.e.
#    +( float, Int64)
#comment: this specifies an add function, whose first parameter is a float ,
# and the second parameter is int64

#or more elegantly , we can write:

# + ( x :: Number , y :: Number ) via promotion

=#

function generator(a :: Number , b :: Number  )

## do .. while
#generate a random number: for a & b
    return generate(a,b)

# generate a number : for a , and another for bir
    Random random = Random()
#TODO: check if valid

end
function fibonacciGenerator( x :: Number, stride :: Number ,seedSample :: Number ,
     pseudoRandomNumberGenerator :: function )

    #generate a number
    generator(x, x + stride)

    pseudoRandomNumberGenerator

end


#Notice:
# 1. The Type of variable x in functions getRandomStart(), getRandomDistance()
# 2. It is the same (i.e. a `Number` )
#But, the logic in each function might be different

# getRandomStart( x :: Number )
function getRandomStart( x :: Number )

end

# getRandomDistance(x :: Number )
function getRandomDistance(x :: Number )

end

# getThreadSafeFibonacciGenerator()
function getThreadSafeFibonacciGenerator()
# in this function
# lock might be required
#check: for function order
    a = getRandomStart()  #  Random();
    c =  getRandomDistance();  #  = Random();
    a,c
end

# DEMO(1):
a,c = getThreadSafeFibonacciGenerator()
fibonacciGenerator(a,c)

# fibonacciGenerator( a :: Number , b :: Number)
function fibonacciGenerator( a :: Number , b :: Number)

    # Calculate Distance c
    # function issues

    #Multithreading : prone to errors
    #Reason:  Liable To a `Racing condition`
    # We cannot do that, right here
#=
    a = getRandomStart()  #  Random();
    c =  getRandomDistance();  #  = Random();
=#

    # Author's original intention
    # A Linear Function b(a,c)
    # b = a + c
            #we do not know that ... yet
            #we do not know that we need to sum 2 numbers ), in this scope [Lazy Scope]
            #Reason: not enough evidence
            #=
            1.There is not enough evidence to that someone might sum both numbers (straight away)
            2.On the contrary: a developer might call another kernel Function )

            =#

    # generator( a , c )
    #Objective (changed): give me a safe function to execute
    getThreadSafeFibonacciGenerator(a, b)

end

""" Calculates the sum  of Fibonacci numbers

    At this scope
    we do not really know how both functions will be added
Questions to ask:

Q1. Are there any threads running in background?
Q2. if so, are they runinng on (1) machine (concurrency issues)
or are they running on (2) or more separate machines  (Real-time processing issues )

"""

# fibonacciSum( a :: Number , b :: Number  )
function fibonacciSum( a :: Number , b :: Number  )

    response =0

    # Valid Case
    if (0 < a And 0 < b )

    #1.base case:

     if ( a == 1  And b == a ) return a + b   end
     elseif ( a > 1 And b > a )

            # under valid case : call a generator
            response = fibonacciSum( fibonacciGenerator(a,b) generator(a) ,  generator(b) )   # fibonacci(a,b)
     end

    # Otherwise:

    else
    # 2. INVALID Input: is given (either a , or b is negative)
    response = -1

 end

    # 3. Return a response (is valid, even without a `return` keyword)
    response
end

fib( a :: Number , b :: Number )

##2.
#= Basic Dispatch

The first function, its first argument: Any:
 1. Unconstrained: can be anything
 2. Accepts different variables & types


The second function has with both Arguments of Abstract type Number
    - we have `Type Heirarchy`
    - we do `Type Annotations` [How?]
    --  by using  `:: Any `

    ---  A return type is `implicit` (when no type is specified)
         But, if want to have more specific, use  Annotations ::

    ---- In Julia:

    1. we don't have (Structural) Inheritance,an idea of sth we may do (in the future) -haven't missed it (so far that much)

    2.Type Definitions: we  define a `Type` Heirarchy, that  we have (2):
       2.1. Abstract Type
       2.2. Concrete Type

    2.1.Abstract Type
    - Like Number & Integers (inherits from Number)
    -- which everything inherits from it
    --(e.g. floating points Floats, Ints (Int64,32,..), Rationals

    2.Concrete Type
    - Which are leaf types
    -- and  have Fields (for specific applications)
    --- (e.g. implementable)
=#

#--- Basic Dispatch
#=
Any Function Can have Different flavors
Where Different Flavors Constitute of  Different `measures`
=#

#= The Different Measures
#1. the fallback function
2. f(a::Number, b::Number)
3. f(a::Number, b)
=#

#  1. f( a::Any , b :: Any ) [Default]
f( a :: Any, b) = "fallback" # a generic fallback

# 2. f( a :: Number ,  b :: Number )
f( a :: Number ,  b :: Number ) = "a and b are both numbers" # both of Type Number

# 3. f( a :: Number , b :: Any)
f( a :: Number , b :: Any) = "a is a number" # second argument isa ::Any

# 4. f( a :: Any , b:: Number )
f( a :: Any , b:: Number ) = "b is a number"

# 5. f( a :: Integer , b :: Integer )
f( a :: Integer , b :: Integer ) = "a and b are both integers" #we could be more specific (constrained)

# DEMO(2)

#1. f ( a::Any , b :: Any ) [Default]
f( "foo", [1, 2]) # : "fallback" -- it compares all functions, none has array explicitly specified, hence  it picks  the `fallback``  (because its 2nd  argument is dynamic ) # Infer 1st: f(a::Any, b) = "fallback"

# 1. f( a::Any , b :: Any ) [Default]

f(1, "bar") #= "a :: number,  b :: String =#

# 2. f( a :: Number ,  b :: Number )
f( 1 , 5.2 ) #= " a and b are both numbers " =#

# 5. f( a :: Integer , b :: Integer )
f(1, 2)     #= a :: Int64 ,  b :: Int64 ; "a,b are both integers" =#


methods(f) #returns all different methods (& its different types function has ) #convenient
#=julia> methods(f)
# 5 methods for generic function "f":
[1] f(a::Integer, b::Integer) in Main at c:\Users\adamus\.git\
JuliaTutorials\JuliaTutorials\Tutorial\LLVM\llvm.jl:32
[2] f(a::Number, b::Number) in Main at c:\Users\adamLutPhi\.git\JuliaTutorials\JuliaTutorials\Tutorial\LLVM\llvm.jl:29
[3] f(a, b::Number) in Main at c:\Users\adamLutPhi\.git\JuliaTutorials\JuliaTutorials
\Tutorial\LLVM\llvm.jl:31[4] f(a::Number, b) in Main at c:\Users\adamus\.git\adamLutPhi\adamLutPhi\Tutorial
\LLVM\llvm.jl:30[5] f(a, b)
in Main at c:\Users\adamus\.git\JuliaTutorials\JuliaTutorials\Tutorial\LLVM\llvm.jl:28

=#

#=--- Diagonal Dispatch/ Parametric Dispatch =#

f{T<:Number}(a::T, b::T) = "a and b are both $(T)s"
f{T<:Number}(a::Vector{T}, b::Vector{T}) = "a and b are both vectors of $(T)s"

#=Cross-reference  Dispatch on tuples of types https://github.com/JuliaLang/julia/issues/10947
# 10947
(Int,) is not a type at all; it's a tuple that contains a type. So you can't use <: on it.

Tuples now "infer" their parameters from constructor arguments (just like other parametric types) :
=#

# Pair{DataType,DataType}
typeof(Int => String) # Pair{DataType,DataType}

# Tuple{DataType,DataType}

typeof((Int, String)) # Tuple{DataType,DataType}

[Int] #1-element Array{DataType,1}: #Int64 # Core.Int64

# Or we can have numbers set, and assign to a new variable array
array = [1 ,2 ,3, 5, 8, 13]

#=
The Difference is that
 Tuples don't provide a way to request parameter types.

In that case, you can let tuples provide parameter types (of your choice):
you can Construct
                        Pair{Any,Any}(1,2)
But you can't do the same thing with tuples.
Tuple "parameters" are always inferred.

This is normally what you'd expect,
but the corner case is that
It's impossible to construct a Tuple{Type{Int}}.

- To do this dispatch,

-- I think you'll have to pass tuple types [How?]
1. dispatch on Type{Tuple{T,T}}.
  Unfortunately the input syntax Tuple{A,B}
  is much less pleasant than (A,B).

2. Another example in favor of using {A,B} for tuple types.

The difference is that:
tuples don't provide a way to request parameter types.
Tuple "parameters" are always inferred.
tuples don't provide a way to request parameter types. If you want, you can construct Pair{Any,Any}(1,2), but you can't do the same thing with tuples. [Why?]
Tuple "parameters" are always inferred.

This is normally what you'd expect, but the corner case is that it's impossible to construct a Tuple{Type{Int}}.

To do this dispatch, I think you'll have to pass tuple types --- dispatch on Type{Tuple{T,T}}. Unfortunately the input syntax Tuple{A,B} is much less pleasant than (A,B)
Another example in favor of using {A,B} for tuple types.
=#

function f(::Tuple{DataType})

end

# isa function


println( isa(Int, Any) )

#true
println( isa(Int, Type{Int}) )
#true
isa(Int, DataType) #true
isa(("",), Tuple{Any}) #true
isa(("",), Tuple{String}) #true
isa(("",), Tuple{ASCIIString}) #true #2022: Error UndefVarErro (is this normal?)
isa((Int,), Tuple{Any}) #true
isa((Int,), Tuple{Type{Int}}) # false
isa((Int,), Tuple{DataType}) #true

#Edit: upon further reflection, I guess I understand
# f typeof((Int,)) is a Tuple{DataType}. And this is behaving like dispatch:

f( ::Tuple{Type{Int}} ) = 1 #f (generic function with 1 method) #works


f((Int,)) #ERROR: MethodError: `f` has no method matching f(::Tuple{DataType}) # Closest candidates are: f(::Tuple{Type{Int64}}) #me : so the ....
#= Note: on the error above: since now compiler  shares with us the expected Type `::Tuple{DataType}` we can extend, and  make out own function =#
# ERROR: MethodError: `f` has no method matching f(::Tuple{DataType}) Closest candidates are: f(::Tuple{Type{Int64}})

#= Warning f(::Tuple{Type{Int64}}) #f(::Tuple{DataType}) #OPEN! #BUGGY ERROR Detected (for more review: src\UI\Gtk\issues\open\isa_issue.jl for more examples of buggy code)
#= Remark:in 2013, pointing to a hidden bug, with a one word f((Int,)) -that breaks Julia itself- was later discovered by chance in 2015; till today the issue is open (because it is as deep as LLVM)
" The presenter is the Least, but the Fellow 'scientists' that he Resemble are" , hereby, spooky=#

#=
 Takeaways:
1. Tuples don't provide a way to request parameter types
2. Tuples now "infer" their parameters from constructor arguments

3. tuples don't provide a way to request parameter types.
4. Tuple "parameters" are always inferred.

Quote:
1.  " The presenter is the Least, but the Fellow 'scientists' that he Resemble are"


=#
