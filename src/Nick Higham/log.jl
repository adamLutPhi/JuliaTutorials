#=
P.4 Complex step Method 

for analytic t i = sqrt(-i)

    f'(x) ≈ Im f(x + i*h)/h 

Error O(h^2)
h not restricted by rounding errors 
can take h = 10^100 

- Approximate automatic Differntiation 
- Alg for f must not emply complex arithmetic

Paper: 
-Lyness & Moler (1967)
- Squire & Trapp (1998)


f(x) = atan(x) / (1+ exp(-x^2)) at x=2:

Prof Higham: "derivative is correct to 15 figures"

Accurate derivative
0.274623728154858 gets all digits correct 

Complex step with h = le-100. #gets all digits correct
0.274623728154858

Forward difference with h = 8*u.
0.250000000000000  

Forward difference with h = sqrt(8*u). 
0.274623729288578

#takeaway: with optimal choice, you get only Half of the digits correct 


"The shortest and best way between two
truths of the real domain often passes
through the imaginary one."
Jacques Hadamard (1945)
log(z) is the Alogorithm for which Im log z E (-1r, 1r]


log(z1 + log(z2)    (9.1)

log z1 * z2 = log(abs(z1*z2)) + i*arg(z1*z2)
            = Log(abs(z1) + Log(z2) + i*arg(z1)+ i*arg(z2)
            = log(z1) + log(z2)

Difference between Log vs log:


book: an introduction to complex analysis R.P Agarwal 2011
turning the page, 9.1 holds if you 
take the appropriate  
"Branch of the log (in each of those occurences of the log )
(not an actual identity to the principal Log)
So, the formula is not , in general, true 
Q.1how can we understand this?
Q2. how can we compute this in practice ?
Goes on say it must take appropriate Brach for 
    each occurence of "log"

Book: NIST Handbook (Olver et al. 2010)
a very widely-used mathematical functions
arcsin, Arccos are "general values" of inverse sine, inverse cosine 
9
4.24(iii) Addition Formulas 

Arcsin(u)  ± Arcsin(v)
= - Arcsin(u*(l - v^2)^1/2 ± v(l - 1t2 )^1/2) 4.24.13
Arccos(u) = Arcos(v) 
= Arccos(u*wi ∓ ((1-u^2))*(1 - u^2)*(1-v^2))^1/2) 4.24.14

Arcsin, Arccos are "general values" of inverse sine, inverse
cosine

Each occuruce of the "Multi-valued function" could be from Different Branch 
Q. if I have a particular u & v 
for my principal branch, I want to know : what can i say
 the Author's Understanding (topic)v.s. Reader Understanding(topic )

 #---P.12. unwinding Number 

 U(z) = z- log(exp(x)^z)/2*pi*i

 Note: log(exp(x)^z)+ 2*pi*i*U(z)
 papers:
 -Corless,Hare & Jeffery (1996)
 -Apostol (1974):special case 

 #---P.13.
 unwinding number is an integer (an alternative formula)

 U(z) = z-log(exp(x)^z)/2*pi*i = ceil(Im(z)-pi/2*pi)

 when is log exp^z = z  #NOT A USUAL ^ 
 U(z) = 0 iff Im(z) ∈ (i*pi,pi} #semi-open interval 
me: reimann surface of logarithm forms a helix variable color(whose value vary)

comment: turns out
1.unwinding number is an integer 
2.reimann Surface: counts which branch of Reimann Surface (you are on)

 when log(e^z)=z  [me: no errors!] #NOT USUAL ^
 it will be as long as imaginary part (Im(z))
 is in a VErtical strip (horizontal strip) - i digress 
 between [-pi,pi]

 #--- Roundtrip Relations 
another identity:
cos(acos(z)) = z [minimal error ]

WARNING: AVOID AMBIGUITY WITH ^ 
SEE:https://github.com/JuliaLang/julia/blob/master/base/complex.jl
there are 2 imaginary number specifications : 
im , imag
 =#

z1 = imag #generic function with 19 methods 
z2 = im::Complex
z1 != z2
z = z2
cos(acos(z)) == z #false 
cos(acos(z)) ≈ z  #true! [already an identity]
# acos(cos(z)) = z # Syntax Error cos(z) is not a valid function 



function log10(z::Complex)
    a = log(z)
    a / log(oftype(real(a), 10))
end
function log2(z::Complex)
    a = log(z)
    a / log(oftype(real(a), 2))
end

function exp(z::Complex)
    zr, zi = reim(z)
    if isnan(zr)
        Complex(zr, zi == 0 ? zi : zr)
    elseif !isfinite(zi)
        if zr == Inf
            Complex(-zr, oftype(zr, NaN))
        elseif zr == -Inf
            Complex(-zero(zr), copysign(zero(zi), zi))
        else
            Complex(oftype(zr, NaN), oftype(zi, NaN))
        end
    else
        er = exp(zr)
        if iszero(zi)
            Complex(er, zi)
        else
            s, c = sincos(zi)
            Complex(er * c, er * s)
        end
    end
end

function expm1(z::Complex{T}) where {T<:Real}
    Tf = float(T)
    zr, zi = reim(z)
    if isnan(zr)
        Complex(zr, zi == 0 ? zi : zr)
    elseif !isfinite(zi)
        if zr == Inf
            Complex(-zr, oftype(zr, NaN))
        elseif zr == -Inf
            Complex(-one(zr), copysign(zero(zi), zi))
        else
            Complex(oftype(zr, NaN), oftype(zi, NaN))
        end
    else
        erm1 = expm1(zr)
        if zi == 0
            Complex(erm1, zi)
        else
            er = erm1 + one(erm1)
            if isfinite(er)
                wr = erm1 - 2 * er * (sin(convert(Tf, 0.5) * zi))^2
                return Complex(wr, er * sin(zi))
            else
                s, c = sincos(zi)
                return Complex(er * c, er * s)
            end
        end
    end
end

function log1p(z::Complex{T}) where {T}
    zr, zi = reim(z)
    if isfinite(zr)
        isinf(zi) && return log(z)
        # This is based on a well-known trick for log1p of real z,
        # allegedly due to Kahan, only modified to handle real(u) <= 0
        # differently to avoid inaccuracy near z==-2 and for correct branch cut
        u = one(float(T)) + z
        u == 1 ? convert(typeof(u), z) : real(u) <= 0 ? log(u) : log(u) * z / (u - 1)
    elseif isnan(zr)
        Complex(zr, zr)
    elseif isfinite(zi)
        Complex(T(Inf), copysign(zr > 0 ? zero(T) : convert(T, pi), zi))
    else
        Complex(T(Inf), T(NaN))
    end
end

function exp2(z::Complex{T}) where {T}
    z = float(z)
    er = exp2(real(z))
    theta = imag(z) * log(convert(float(T), 2))
    s, c = sincos(theta)
    Complex(er * c, er * s)
end

function exp10(z::Complex{T}) where {T}
    z = float(z)
    er = exp10(real(z))
    theta = imag(z) * log(convert(float(T), 10))
    s, c = sincos(theta)
    Complex(er * c, er * s)
end

# _cpow helper function to avoid method ambiguity with ^(::Complex,::Real)
function _cpow(z::Union{T,Complex{T}}, p::Union{T,Complex{T}}) where {T}
    z = float(z)
    p = float(p)
    Tf = float(T)
    if isreal(p)
        pᵣ = real(p)
        if isinteger(pᵣ) && abs(pᵣ) < typemax(Int32)
            # |p| < typemax(Int32) serves two purposes: it prevents overflow
            # when converting p to Int, and it also turns out to be roughly
            # the crossover point for exp(p*log(z)) or similar to be faster.
            if iszero(pᵣ) # fix signs of imaginary part for z^0
                zer = flipsign(copysign(zero(Tf), pᵣ), imag(z))
                return Complex(one(Tf), zer)
            end
            ip = convert(Int, pᵣ)
            if isreal(z)
                zᵣ = real(z)
                if ip < 0
                    iszero(z) && return Complex(Tf(NaN), Tf(NaN))
                    re = power_by_squaring(inv(zᵣ), -ip)
                    im = -imag(z)
                else
                    re = power_by_squaring(zᵣ, ip)
                    im = imag(z)
                end
                # slightly tricky to get the correct sign of zero imag. part
                return Complex(re, ifelse(iseven(ip) & signbit(zᵣ), -im, im))
            else
                return ip < 0 ? power_by_squaring(inv(z), -ip) : power_by_squaring(z, ip)
            end
        elseif isreal(z)
            # (note: if both z and p are complex with ±0.0 imaginary parts,
            #  the sign of the ±0.0 imaginary part of the result is ambiguous)
            if iszero(real(z))
                return pᵣ > 0 ? complex(z) : Complex(Tf(NaN), Tf(NaN)) # 0 or NaN+NaN*im
            elseif real(z) > 0
                return Complex(real(z)^pᵣ, z isa Real ? ifelse(real(z) < 1, -imag(p), imag(p)) : flipsign(imag(z), pᵣ))
            else
                zᵣ = real(z)
                rᵖ = (-zᵣ)^pᵣ
                if isfinite(pᵣ)
                    # figuring out the sign of 0.0 when p is a complex number
                    # with zero imaginary part and integer/2 real part could be
                    # improved here, but it's not clear if it's worth it…
                    return rᵖ * complex(cospi(pᵣ), flipsign(sinpi(pᵣ), imag(z)))
                else
                    iszero(rᵖ) && return zero(Complex{Tf}) # no way to get correct signs of 0.0
                    return Complex(Tf(NaN), Tf(NaN)) # non-finite phase angle or NaN input
                end
            end
        else
            rᵖ = abs(z)^pᵣ
            ϕ = pᵣ * angle(z)
        end
    elseif isreal(z)
        iszero(z) && return real(p) > 0 ? complex(z) : Complex(Tf(NaN), Tf(NaN)) # 0 or NaN+NaN*im
        zᵣ = real(z)
        pᵣ, pᵢ = reim(p)
        if zᵣ > 0
            rᵖ = zᵣ^pᵣ
            ϕ = pᵢ * log(zᵣ)
        else
            r = -zᵣ
            θ = copysign(Tf(π), imag(z))
            rᵖ = r^pᵣ * exp(-pᵢ * θ)
            ϕ = pᵣ * θ + pᵢ * log(r)
        end
    else
        pᵣ, pᵢ = reim(p)
        r = abs(z)
        θ = angle(z)
        rᵖ = r^pᵣ * exp(-pᵢ * θ)
        ϕ = pᵣ * θ + pᵢ * log(r)
    end

    if isfinite(ϕ)
        return rᵖ * cis(ϕ)
    else
        iszero(rᵖ) && return zero(Complex{Tf}) # no way to get correct signs of 0.0
        return Complex(Tf(NaN), Tf(NaN)) # non-finite phase angle or NaN input
    end
end

#=Roundtrip Relations 
Theorem 

if Re z != k*pi ; k \in Z then 


    acos(cos(z)) #not existant per se 
    = z - 2*pi*U(i*z)*sign(z-2*pi*U(i*z)))
Corollary 
acos(cos(z)) = z iff Re z z∈(0,pi)

=#
#=Accurate Difference

how to compute Accurate Value of 
log \lambda , λ1,λ2 ∈ C
=#
# λ1, λ2 = #rand(im,1)# no rand for complex 
#unwinding number 
U(z) = (z - log(exp(z))) / 2 * pi * im # takeaway:the use of generic functon promotes lazy processing - easy on memory, fast in response 

λ1 = 0.75;
λ2 = 0.25;
lhs = log(λ2) - log(λ1)

z = (λ2 - λ1) / (λ2 + λ1)

#TODO: check / Recheck is required 
#z = log(exp(z)) + 2 * pi * im * U(z) #no method matching exp(::Float64)
#=
log(λ2) - log(λ1) 
someway, i can rewrite the expression , for more accurate Evaluation 
by introducing the z:
#TODO: check error1 = abs(lhs - rhs1) 
=#

rhs1 = log(λ2 / λ1) + 2 * pi * im * U(log(λ2) - log(λ1))# my bad: U(z) * (log(λ2) - log(λ1) # now, it's corrrect #checked

rhs1_5 = log((1 + z) / (1 - z)) + 2 * pi * im * U(log(λ2) - log(λ1)) #this value is also sound 
#result clustering rhs1, rhs1_5 
error_initial = rhs1 - rhs1_5 # error is the same  (same values ) # logical error: comparing 2 RHSs # TODO: compare each RHS with its respective LHS

error1 = abs(rhs1) - abs(lhs) # 2.220446049250313e-16 < \epsilon≈ 0

#= "The Difference of Logs as a log Ratio
problem: need a correction term 
solution: that's where the unwinding number steps in  "
=log((1+z)/(1-z) + 2*π*i*u(log(λ2)-log(λ1))


=assuming we have a good atanh function 
=#
rhs2 = atanh(z) + 2 * pi * im * U(log(λ2) - log(λ1)) #used in state of art matrix log codes 

#=you can be sure that won't give you an accurate estimation 
(even if lambdas are close )
used in state of the art matrix log codes 

emphasize: z still has λ2 - λ1 (in it)
(but they're subtracting original numbers, that was given)

=#

error = abs(z - rhs2)
#--- the Significance (& the Plateau)
#= one value isn't & won't be enough for correctly identifying the Significance of values
    Thus, it is in my pure interest to develop a 'Gray Area' around 0.05 
    My prine concern is to build a more valuable plateau around the Significant value (rather than relying on an arbitrary line in sand)
    This would render results as one of three: Significant, insignificant, indeterminant 
    The hardest thing is failing to admit our indeterminance 
    Our determenance is defined by the thickness of the indeterminance, the plateau must always be smaller than the Significance itself
    Let's say we have a line of Significance at 0.05 , then there must be a free-zone revolving around it, that is equal to each other- as above, so below 
    0.05 ± 0.01 : 0.01 thus, absolute thickness would make up 0.02 

    =#
isSignificatant = error < 0.05 == true || error >= 0.05 == false
