"""MethodAnalysis Tutorial 
Credits go to:
Tim Holy -  youtube
(as a Part of Series of High-Performance Computing Series)
 
# Pearl1 
only bit of code, defining method(add2): Doing different things when given differnt Inputs (me: Julia helps that by automatic allocation (of data types?))

# Pearl2 
julia adds this method the list, upon reading this new DataType Input(Vector{Int64} (alias for Array{Int64, 1})) -returns final answer (as all variables initialized)

 """
addTwo(x) = x[1] + x[2]  # Pearl1 : MethodInstance for addTwo(::Vector{Int64}) 
typeof(addTwo(x)) #thought this suffice ##ERROR: need variable to be defined- methodinstances to use instead
#######################
addTwo([1.0, 2.0]) # Pearl2 2-element Vector{Core.MethodInstance}: MethodInstance for addTwo(::Vector{Int64}) MethodInstance for addTwo(::Vector{Float64})

addTwo([1, 2])


"""
```
One way to Produce multiple Appetites, is to Produce Multiple Methods - Tim Holy
```

running methods tells us that 
how many Number of Methods
"""

methods(addTwo)
"""to Extract the Method (to be called) giveen types (of Argument) 
Julia Defines (at) which macro, so that you can 
"""
m = @which addTwo([1, 2]) # same pronoun

function whichMacro(fun = addTwo([1, 2]))
    return @which addTwo([1, 2])
end

using MethodAnalysis
"""
A package with:

methodinstances(@which add())
"""

methodinstances(@which addTwo([1, 2]))


# For DataType ( int or  float) #- its just a 2 ways of "mthod specization "

@code lowered add2([1, 2]) #- represents the method, not a particular mrthodinstance

"""

one method 2 or more method specialization 



you Can create methods, that depend on specific types


that is one created by a compiler a.k.a **compiler specialization**
me: see- multidispatching
#----

Let's make another

"""

addTwo((1, 2.0))


methodinstances(whichMacro(addTwo((1, 2.0))))
"""
3-element Vector{Core.MethodInstance}:
 MethodInstance for addTwo(::Vector{Int64})
 MethodInstance for addTwo(::Vector{Float64})
 MethodInstance for addTwo(::Tuple{Int64, Float64})

Redundancy-free

- Julia creates these first time ,call add2 with a new type 

- On second call(later call), julia just uses the code it has already compiled

all compilation: the ammount of work to be done to compile all these methods & instances 
can actually take long-time 

sinc it has the code, in the running session , when you call function 

'''if already done (julia allocated session with this function in memory)
then let's just run that 

"""

#---what differences betwen MethodInstancees 
#=
running macro @code 

    macro code_lowered(ex0...)
    thecall = gen_call_with_extracted_types_and_kwargs(__module__, :code_lowered, ex0)
    quote
        results = $thecall
        length(results) == 1 ? results[1] : results
    end
end
=#

"""
gives a representation of a specific method 
that is applicable


CodeInfo(
1 ─ %1 = Base.getindex(x, 1) x[1] implemented as a call `getindex(x,1)` (defined in Base)
│   %2 = Base.getindex(x, 2)
│   %3 = %1 + %2
└──      return %3
)

"""
@code_lowered addTwo([1, 2])


#=

%3 = %1 + %2 
are the compiler lingo, some other markers 
indicate blocks of code that execute
codeblock:
executes without branches (no if,while,... within the block)

why? the reson is, no additional specialization
look to now at mthod instances 
how? by Switching to Another Macro

note:
optimize = false : turns off some julia optimization 
(Inference Engine) is willing to make
=#

"""addTwo 
specialized for whatever type its Arguments are

    code_typed optimize
Like code_lowered with type annotations 

"""
@code_typed optimize = false addTwo([1, 2]) #represents the specific methodinstances
"""addTwo 
specialized for whatever type its Arguments are

    code_typed optimize


    @code_typed optimize = false addTwo([1, 2])
CodeInfo(
1 ─ %1 = Base.getindex(x, 1)::Int64
│   %2 = Base.getindex(x, 2)::Int64
│   %3 = (%1 + %2)::Int64
└──      return %3
) => Int64
"""

@code_typed optimize = false addTwo([1.0, 2.0])

"""
CodeInfo(
1 ─ %1 = Base.getindex(x, 1)::Float64
│   %2 = Base.getindex(x, 2)::Float64
│   %3 = (%1 + %2)::Float64
└──      return %3
) => Float64
"""
#--- mixed input : 1 int, 2 float 
@code_typed optimize = false addTwo((1, 2.0))

"""

CodeInfo(
1 ─ %1 = Base.getindex(x, 1)::Int64
│   %2 = Base.getindex(x, 2)::Float64
│   %3 = (%1 + %2)::Float64
└──      return %3
) => Float64
"""
#--- how julia knows about this: Type Inference

typeof([1, 2])


@code_typed optimize = false getindex([1, 2], 1)
#=
CodeInfo(
1 ─ %1 = Base.arrayref($(Expr(:boundscheck)), A, i1)::Int64
└──      return %1
) => Int64
=#
@code_typed optimize = false 1 + 2

#=
CodeInfo(
1 ─ %1 = Base.mul_int(x, y)::Int64
└──      return %1
) => Int64

=#

"""julia is built on top of a set of v. primitive operations 
that ar essentially built-in 
add_int is a primitive , built-in (not a generic type)
-it's one of the 80 (or so) primitives that julia is built on  
"""

@code_typed optimize = false Base.add_int(1, 2) # 1 * 2 #cannot further introspect



"""
arametrization.jl:213

CodeInfo(
1 ─ %1 = Base.promote(x, y)::Tuple{Float64, Float64} # promotes the first 2 different types 
│   %2 = Core._apply_iterate(Base.iterate, Base.:+, %1)::Float64 # calls plus 
└──      return %2
) => Float64

Note: same syntax for a variable argument - multiple arguments passed into the same function 

that is inefficient, but julia optimizes that thing 

how by turning optimization feature of code_typed

CodeInfo(
1 ─ %1 = Base.sitofp(Float64, x)::Float64
│   %2 = Base.add_float(%1, y)::Float64
└──      return %2
) => Float64

converts integer to Float64
2 then it adds 2 Float64

"""

@code_typed optimize = true 1 + 2.0


#--- llvm 

"""
Note: this is as far as you go to Diagnose your problems 

julia also converts code deeper to
 llnm 
format that interacts with compiler using macros 

@code_llvm

@code_native  - the final CPU  instructions, optimized for your specific Machine

"""
s = nothing
"""
code llvm 

@code_llvm optimize = false 1 + 2 

;  @ int.jl:87 within `+'
; Function Attrs: uwtable
define i64 @"julia_+_2061"(i64 signext %0, i64 signext %1) #0 {
top:
  %2 = call {}*** @julia.ptls_states()
  %3 = bitcast {}*** %2 to {}**
  %4 = getelementptr inbounds {}*, {}** %3, i64 4
  %5 = bitcast {}** %4 to i64**
  %6 = load i64*, i64** %5, align 8
  %7 = add i64 %0, %1
  ret i64 %7
}
"""
@code_llvm optimize = true 1 + 2


#=
;  @ promotion.jl:321 within `+'
; Function Attrs: uwtable
define double @"julia_+_2096"(i64 signext %0, double %1) #0 {
top:
  %2 = call {}*** @julia.ptls_states()
  %3 = bitcast {}*** %2 to {}**
  %4 = getelementptr inbounds {}*, {}** %3, i64 4
  %5 = bitcast {}** %4 to i64**
  %6 = load i64*, i64** %5, align 8
; ┌ @ promotion.jl:292 within `promote'
; │┌ @ promotion.jl:269 within `_promote'
; ││┌ @ number.jl:7 within `convert'
; │││┌ @ float.jl:94 within `Float64'
      %7 = sitofp i64 %0 to double
; └└└└
;  @ promotion.jl:321 within `+' @ float.jl:326
  %8 = fadd double %7, %1
;  @ promotion.jl:321 within `+'
  ret double %8
}

-----
;  @ promotion.jl:321 within `+'
; Function Attrs: uwtable
define double @"julia_+_2094"(i64 signext %0, double %1) #0 {
top:
; ┌ @ promotion.jl:292 within `promote'
; │┌ @ promotion.jl:269 within `_promote'
; ││┌ @ number.jl:7 within `convert'
; │││┌ @ float.jl:94 within `Float64'
      %2 = sitofp i64 %0 to double
; └└└└
;  @ promotion.jl:321 within `+' @ float.jl:326
  %3 = fadd double %2, %1
;  @ promotion.jl:321 within `+'
  ret double %3
}
=#

@code_llvm optimize = false 1 + 2.0

#=@code_native 1 + 2
        .text
; ┌ @ int.jl:87 within `+'
        pushq   %rbp
        movq    %rsp, %rbp
        leaq    (%rcx,%rdx), %rax
        popq    %rbp
        retq
        nopw    (%rax,%rax)
; └

julia> ) => Float64

//---------------------------
julia> @code_native 1 + 2.0
        .text
; ┌ @ promotion.jl:321 within `+'
        pushq   %rbp
        movq    %rsp, %rbp
; │┌ @ promotion.jl:292 within `promote'
; ││┌ @ promotion.jl:269 within `_promote'
; │││┌ @ number.jl:7 within `convert'
; ││││┌ @ float.jl:94 within `Float64'
        vcvtsi2sd       %rcx, %xmm0, %xmm0
; │└└└└
; │ @ promotion.jl:321 within `+' @ float.jl:326
        vaddsd  %xmm1, %xmm0, %xmm0
; │ @ promotion.jl:321 within `+'
        popq    %rbp
        retq
        nop
; └

=#


#= @code_native addTwo(1, 2)

as it is a generic function , There is no matching native code for it 
ERROR: LoadError: no unique matching method found for the specified argument 
types
=#


#= @code_llvm addTwo(1, 2)
ERROR: LoadError: no unique matching method found for the specified argument 
types
=# #uncomment line for testing only
#@code_llvm addTwo(1, 2)




