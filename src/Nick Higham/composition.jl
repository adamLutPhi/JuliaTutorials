#--- Chain Rule Nick Higaham - @nickhigham

#=
    function count =
=#
# 1. g(x): loop(of Jacobians)
# g'(x) = Jf(x) : the  `Jacobian Matrix`

# Chain Rule
g(x)=( f3 * ( f2(x) f1(x) )

#=

Chain rule gives...  #TODO: a loop on allJacobians


Suppose you have a function g(of x) -

1.a function of Composition
2. of other functions: f3, f2, f1)
3. each f maps some Vectors (of length n [k] -to-> n [ k+1 ] )

1.so the
(the 1st) Derivative of g : is a  `Jacobian Matrix`

2.if Apply the chain Rule:
I'll get the expression (Jf):

Note: the Product of 3 Jacobians
Evaluated at the `Appropriate Arguments`


Q1.If I want to get the Jacobian (J)
 - In which order should I take
  J3 J2 J1 or J1, J2, J3 ?
A1.
 It comes up in `Automatic Differentiation AD` :
  the difference between the first mode

Reverse mode (output) & forward (input )

So if i want to get the derivative of g

#-- Testing Area:


for i in enumerate(n)

  if i < 2
    return jf(i,)
    end
end

=#
for i in enumerate(n)

  if i < 2
    return jf(i,)
    end
end

#Composition (of functions )

Jg(x) =  Jf3(f2 * (f1(x))) * Jf2(f1(x)) *Jf1(x) ;

#---Todo implement Jacobian too

#=y Symbolics.substitute
https://discourse.julialang.org/t/how-to-evaluate-substitute-numerical-values-of-params-n-states-a-jacobian-in-modelingtoolkit-using-generate-jacobian-or-calculate-jacobian/71466

Solution
yewalenikhil65
Nov 2021
@Credits: @baggepinnen
I am doing currently following

states , parameters
i == 1
x1map =

Warning: ModelingToolkit uses `ZygoteRules`

Plus next line it uses Symbolics too writer seems
Can we do better ?

@Credits gladnde

support SVector but it isn’t managing the conversion.

function BicycleDynamics(plant::Bicycle, x::SVector{5, T}, u::SVector{2, T}) where {T <: Real}
    # of the traction wheel
    turning_radius = plant.wheelbase / tan(x[4])
    translational_velocity = x[5]
    angular_velocity = x[5] / turning_radius
    return SVector(cos(x[3]), sin(x[3]), angular_velocity, u[1], u[2])
end

bike = MakeBike()

ad_adapter(x::SVector{7, T}) where {T <: Real} = BicycleDynamics(bike, x[1:5], x[6:7])
ad_point = SVector{7, Float64}(1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0)

#Warning: uses `ForwardDiff` package
ForwardDiff.jacobian(ad_adapter, ad_point)


@Credit

To expand on this, if what someone wants a callable function, j, that computes the Jacobian then this should work:

julia> f(x,y)=[x^2 + y^3-1, x^4 - y^4+ x*y]
julia> j(x)=ForwardDiff.jacobian(x->f(x[1],x[2]), x)
julia> j([0.5;0.5])
2x2 Array{Float64,2}:

@Credits jchen975 (May 2020)
=#

""" Functors
@Credit: GeeksforGeeks.com
"Functors, also known as
function objects, are objects that behave like functions"
treated as a function, can be used to encapsulate a set of instructions

"""
function f((x))


    F = zeros(3)

    F[1] = x[1]^2 + x[3]
    F[2] = x[1] + x[2]
    F[3] = x[2]^2 + x[3]^2

    return F
end

#=
@Credits yewalenikhil65 (Nov 2021)
Requires ModelingToolkit, Symbolics, Zygote
x₀map = states(odesys) .=> x0

pmap  = parameters(odesys) .=> ps
jac = substitute.( ModelingToolkit.calculate_jacobian(odesys), (Dict([x₀map;pmap]),) )
jac = Symbolics.value.(jac)

pmap = nothing ;jac=nothing ;x1=0;ps =;
x₀map = states(odesys) .=> x1
pmap  = parameters(odesys) .=> ps
jac = substitute.( ModelingToolkit.calculate_jacobian(odesys), (Dict([x₀map;pmap]),) )
jac = Symbolics.value.(jac);

@baggepinnen (Nov 2021)
#requires Symbolics
Try Symbolics.substitute

=#

#=the key:
You only need function f(x)

Defining F as zeros(3) restricts its entries to be Float64s,
Whereas to use `ForwardDiff`
They need to be `Duals`.

Change it to F = similar(x) or F = zeros.(x)
#source: https://discourse.julialang.org/t/jacobian-of-a-multivariate-function/21131/10
=#

#=@Credits: dpsanders
dpsanders
You only need function f(x)

Defining F as zeros(3) restricts its entries to be Float64s, whereas to use ForwardDiff
they need to be Duals.

Change it to F = similar(x) or F = zeros.(x)

to get something of the correct type:

function f(x)
    F = zeros.(x)

    F[1] = x[1]^2 + x[3]
    F[2] = x[1] + x[2]
    F[3] = x[2]^2 + x[3]^2

    return F
end
to get something of the correct type: =#

"""https://discourse.julialang.org/t/jacobian-of-a-multivariate-function/21131/10
Change it to F = similar(x) or F = zeros.(x)

to get something of the correct type:
"""
function f(x)

    F = zeros.(x)

    F[1] = x[1]^2 + x[3]
    F[2] = x[1] + x[2]
    F[3] = x[2]^2 + x[3]^2

    return F
end


#--- Threads

using Base.Threads
nthreads()
@threads for i = 1:6
    println("Hello from thread: ", T)
#--- chain rule gives
#=
function _f(x)
  F = similar(x) #zeros.(x)#small perturbation

  F[1] = x[1]^2 + x[3] # function here
  F[2] = x[1] + x[2] #another function
  F[3] = x[2]^2 + x[3]^2 #third one too

  return F
end
=#
#=@Credits carstenbauer (Jan 13th)
> export JULIA_NUM_THREADS=6
> julia

or
JULIA_NUM_THREADS = 6
julia

or
JULIA_NUM_THREADS = 6
=#
JULIA_NUM_THREADS = 6

#--- Jacobians

g(x) = f3( f2(f1(x) ))   # fk R^n => R^nk+1;

Jg(x) = Jf3( f2( f1(x) )) * Jf2(f1( x )) * Jf1( x );

#=
Q. does Jg as (Jf3*Jf2)*Jf1 or Jf3*(Jf2*(Jf1))
me: i.e. is it forward or backward (see also: nick-Higham juliacon 2018)

https://github.com/JuliaDiff/SparseDiffTools.jl
In addition, the following forms allow you to provide a gradient function g(dy,x) or dy=g(x) respectively:
=#
num_hesvecgrad!(dy,g,x,v,
                     cache2 = similar(v),
                     cache3 = similar(v))

num_hesvecgrad(g,x,v)

auto_hesvecgrad!(dy,g,x,v,
                     cache2 = ForwardDiff.Dual{DeivVecTag}.(x, v),
                     cache3 = ForwardDiff.Dual{DeivVecTag}.(x, v))

auto_hesvecgrad(g,x,v);

#---
#The `numauto`and autonum methods both mix numerical and automatic differentiation, with the former almost always being more efficient and thus being recommended.

# Optionally, if you load Zygote.jl, the following numback and autoback methods are available and allow numerical/ForwardDiff over reverse mode automatic differentiation respectively, where the reverse-mode AD is provided by Zygote.jl. Currently these methods are not competitive against numauto, but as Zygote.jl gets optimized these will likely be the fastest.

#using Zygote # Required #i'm not sure # are there anytthing else?

numback_hesvec!(dy,f,x,v,
                     cache1 = similar(v),
                     cache2 = similar(v))

numback_hesvec(f,x,v)

# Currently errors! See https://github.com/FluxML/Zygote.jl/issues/241 #says about Zygote
autoback_hesvec!(dy,f,x,v,
                     cache2 = ForwardDiff.Dual{DeivVecTag}.(x, v),
                     cache3 = ForwardDiff.Dual{DeivVecTag}.(x, v))

autoback_hesvec(f,x,v)
#=
Jv and Hv Operators
The following produce matrix-free operators which are used for calculating Jacobian-vector and Hessian-vector products where the differentiation takes place at the vector u:
=#
JacVec(f,x::AbstractArray;autodiff=true)
HesVec(f,x::AbstractArray;autodiff=true)
HesVecGrad(g,x::AbstractArray;autodiff=false)

#These all have the same interface,
# Whereas J*v utilizes the out-of-place Jacobian-vector or Hessian-vector function,
#whereas mul!(res,J,v) utilizes the appropriate in-place versions.
# To update the location of differentiation in the operator,
# simply mutate the vector u: J.u .= ....
#mutating vector u:
u: J.u
#or
mul!(res, J, v)


#= Chain Rule Modes of differentiation - Prof. Nick Higham

 1.Automatic Differentiation (AD)
 2.Forward mode
 3.Reverse mode

Reference:
G.Strang Linear Algebra & Learning from Data
Wellesley-Cambridge Press 2019
see also: @Credits: Gilbert Strang - Linear Algebra and Learning From Data (2019)
https://math.mit.edu/~gs/learningfromdata/

#=
Wellesley-Cambridge Press 2019
=#
=#
rowl = [2 2 3; 3 4 7]'
cols= [2 2 3 ; 3 4 7]'
x1 = [2 2 3]'
x2 = [3 4 7]'
