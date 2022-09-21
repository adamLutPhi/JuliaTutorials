#--- Chain Rule


g(x)=(f3 *(f2(x)f1(x)) 
#=
chain rules give  #TODO: a loop on allJacobians 

@nickhigham:
suppose you have a function g(of x) - 
    a function of Composition 
  (of other functions: f3, f2, f1)      
  each f maps Vectors (of length nk -to-> nk+1)
1.so the  
(1st) derivative of g = Jacobian Matrix 
2.if Apply the chain Rule:
I'll get the expression (Jf):
the product of 3 Jacobians 
Evaluated at the appropriate arguments

Question: 
if I want to get the Jacobian (J)
  which order should I take 
  J3 J2 J1 or J1, J2, J3 ?

  comes up in automatic differentiation:
  the difference between the first mode 

Reverse mode (output) & forward (input )


 
so if i want to get the derivative of g 

#-- testing Area:

for i in enumerate(n)

  if i <2 
    return jf(i,)

=#
#Composition
Jg(x) =  Jf3(f2*(f1(x)))*Jf2(f1(x))*Jf1(x) ;

#---Todo implement Jacobian too     

#=y Symbolics.substitute
https://discourse.julialang.org/t/how-to-evaluate-substitute-numerical-values-of-params-n-states-a-jacobian-in-modelingtoolkit-using-generate-jacobian-or-calculate-jacobian/71466

Solution 
yewalenikhil65
Nov 2021 
thanks @baggepinnen 
i am doing currently following

states ,parameters
i == 1
x1map = 

warning: ModelingToolkit uses ZygoteRules
plus next line it uses Symbolics too writer seems 
can we do better ? 
=# 
#= Requires ModelingToolkit, Symbolics, Zygote (3 in 1) 
pmap = nothing ;jac=nothing ;x1=0;ps =;
x₀map = states(odesys) .=> x1
pmap  = parameters(odesys) .=> ps
jac = substitute.( ModelingToolkit.calculate_jacobian(odesys), (Dict([x₀map;pmap]),) )
jac = Symbolics.value.(jac);
=#
#=the key
You only need function f(x)

Defining F as zeros(3) restricts its entries to be Float64s, whereas to use ForwardDiff
they need to be Duals.

Change it to F = similar(x) or F = zero.(x)
#https://discourse.julialang.org/t/jacobian-of-a-multivariate-function/21131/10
=#

#=@dpsanders 

to get something of the correct type: =#




#--- chain rule gives 
#=
function _f(x)
  F = similar(x) #zero.(x)#small perturbation

  F[1] = x[1]^2 + x[3] # function here
  F[2] = x[1] + x[2] #another function 
  F[3] = x[2]^2 + x[3]^2 #third one too

  return F
end
=#
g(x) = f3(f2(f1(x)))   # fk R^n => R^nk+1;

Jg(x) = Jf3(f2(f1(x)))*Jf2(f1(x))*Jf1(x);

#=
Q. does Jg as (Jf3*Jf2)*Jf1 or Jf3*(Jf2*(Jf1)) 
me: i.e. is it forward or backward (check also nick-Higham juliacon 2018)

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

#These all have the same interface, where J*v utilizes the out-of-place Jacobian-vector or Hessian-vector function, whereas mul!(res,J,v) utilizes the appropriate in-place versions. To update the location of differentiation in the operator, simply mutate the vector u: J.u .= ....

#=Chain Rule Modes of differentiation - Prof. Nick Higham 

 1.Automatic Differentiation (AD) 
 2.Forward (mode)
 3.Reverse mode

Provides a  references 
G.Strang Linear Algebra & Learning from Data 
Wellesley-Cambridge Press 2018 
http://math.mit.edu/~gs/Learningfromdata 

#Final inference:
no body has a clear idea about Automatic differentiation - and is still 
a ripe  area of Research  

"No body knows anything about anything"

=#