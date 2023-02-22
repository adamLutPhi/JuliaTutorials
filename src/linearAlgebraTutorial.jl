module linearAlgebraTutorial
using LinearAlgebra: dot, cross, factorize, cholesky, Diagonal , lowrankdowndate  , lowrankupdate!,lu!  #cholfact! 
using Test, Random
import Base: eps, copy # import: fixes "LoadERROR: must be explicitly imported to be extended"
#eps(::Vector{Float64}) where T<:AbstractFloat #eps() # www.web.mit./doc/julia/html/stdlib/base
using Base: @inbounds #abs #  no method matching abs(::Vector{Float64})
using LinearAlgebra
using LinearAlgebra: issymmetric, isposdef # is positive definite
#using SparseMatrixDicts: sparse #works  but maybe [not for dedicated matrices] 
using SparseMatricesCSR: SparseMatrixCSR  # Suitable
using SparseMatricesCSR: symsparsecsr
# using SparseMatricesCSR: sparsecsr, symsparsecsr # found (incompatible with SparseArrays')
using SuiteSparse
using SparseArrays: sparse # (not in LinearAlgebra) #sparsecsr #not-found in registry 

import SparseArrays: nnz, findnz, dimlub # `dimlub` must be explicitly imported to be extended
# _rows, _cols, _vals, n, n
  #source: test/SymSparseMatrixCSR.jl https://github.com/gridap/SparseMatricesCSR.jl/search?q=I_up&type=
function test_csr(Bi=1,Tv=[1,2,3],Ti=[1,1,1])
 n =3 
 maxrows = n; maxcols = n
  Bi = 1
  #tv=3
  offset = Bi - 1 
  maxnz=10


   maxrows=5
   maxcols=6
  
   J = Ti[1,1,2,2,3,2,5,5]
   V = Tv[2,3,3,4,4,7,7,8]

   I_up = Ti[1,1,2,2,5]
   J_up = Ti[1,2,3,5,5]
   V_up = Tv[2,3,4,7,8]

 I_up, J_up, V_up
end

 """
    sparsecsr(args...)
    sparsecsr(::Val{Bi},args...) where Bi
Create  a `SparseMatrixCSR` with `Bi`-based indexing (1 by default)
from the same `args...` as one constructs a `SparseMatrixCSC`
with the [`sparse`](@ref) function.
"""
sparsecsr(I,J,V) = SparseMatrixCSR(transpose(sparse(J,I,V,dimlub(J),dimlub(I))))
sparsecsr(I,J,V,m,n) = SparseMatrixCSR(transpose(sparse(J,I,V,n,m)))
sparsecsr(I,J,V,m,n,combine) = SparseMatrixCSR(transpose(sparse(J,I,V,n,m,combine)))
sparsecsr(::Val{Bi},I,J,V) where Bi = SparseMatrixCSR{Bi}(transpose(sparse(J,I,V,dimlub(J),dimlub(I))))
sparsecsr(::Val{Bi},I,J,V,m,n) where Bi = SparseMatrixCSR{Bi}(transpose(sparse(J,I,V,n,m)))
sparsecsr(::Val{Bi},I,J,V,m,n,combine) where Bi = SparseMatrixCSR{Bi}(transpose(sparse(J,I,V,n,m,combine)))

#Demo Sparse matrix 
Bi=1; Tv = Vector{Int64};Ti=[1,1,1];   # Tv=[1,2,3];
Tv= eltype(Tv) # Int64
maxrows = length(Ti)
println(typeof(Tv) )
#1
Bi = 1
y = Vector{Tv}(undef,maxrows)
z = Vector{Tv}(undef,maxrows)

_rows = rand(Ti[1]:Ti[maxrows],maxnz)
_cols = rand(Ti[1]:Ti[maxcols],maxnz)
_vals = rand(Tv,maxnz)
function Tva()
    Tv = eltype(V)
    α = Tv(0.5)
    for k in 1:length(I)
      r = I[k]
      c = J[k]
      if r > c
        I[k] = c
        J[k] = r
      end
      if r != c
        V[k] = α*V[k]
      end
    end
  end
  SymSparseMatrixCSR(sparsecsr(Val(Bi),I,J,V,args...))
end
  I = Ti[1,2,1,3,2,5,2,5]
  J = Ti[1,1,2,2,3,2,5,5]
  V = Tv[2,3,3,4,4,7,7,8]

  I_up = Ti[1,1,2,2,5]
  J_up = Ti[1,2,3,5,5]
  V_up = Tv[2,3,4,7,8]
  mul!(y,CSR,x)
  mul!(z,CSC,x)
  @test y ≈ z

  mul!(y,CSR,x,1,2)
  mul!(z,CSC,x,1,2)
  @test y ≈ z

  @test CSR*x ≈ CSC*x

  out = LinearAlgebra.fillstored!(CSR,3.33)
  @test out === CSR
  LinearAlgebra.fillstored!(CSC,3.33)
  mul!(y,CSR,x)
  mul!(z,CSC,x)
  @test y ≈ z
 
 
#----
x = rand(Tv,maxcols)
y = Vector{Tv}(undef,maxrows)
z = Vector{Tv}(undef,maxrows)
mul!(y,CSR,x)
mul!(z,CSC,x)
@test y ≈ z
@test CSR*x ≈ CSC*x
#----
csr = symsparsecsr(I_up,J_up,V_up)
test_csr(Bi,Tv,Ti)

_rows,_cols,_vals = findnz(csr)
# sparsecsr
#2
 CSR = sparsecsr(Val(Bi),_row,_col,_vals,maxrows,maxcols)
 
_rows,_cols,_vals = findnz(CSR)
#L = _rows ; M = _cols; N = _vals; 
csr = sparsecsr(Val(Bi),i,j,v,maxrows,maxcols)
CSC = sparse(_rows,_cols,_vals)
  if Bi == 1
    CSR = sparsecsr(I,J,V)
    @test CSR == CSC
  end
  
  function getDims(x)
   _rows =  size(x, 1)
   _cols = size(x, 2)
   _vals = size(x, 3)
  _rows, _cols, _vals
  end 
   #_rows, _cols, _vals = getDims(x)
  
  #sparse(_rows,_cols,_vals)
  #lambdas 
  sparsecsr(_rows,_cols,_vals) = SparseMatrixCSR(transpose(sparse(_rows,_cols,_vals,dimlub(_rows),dimlub(_cols))))
 #sparsecsr(L,M,N) = SparseMatrixCSR(transpose(sparse(J,I,V,dimlub(J),dimlub(I))))
 
 
function sparseMatrix(_value, x,_rows,_cols,_vals, args...)
   sparseMat = sparsecsr(_value(x),_rows,_cols,_vals,args...)
   sparseMat
end

#=
#function luDecompose!()
function LinearAlgebra.lu!(
  translu::Transpose{T,<:SuiteSparse.UMFPACK.UmfpackLU{T}},
  a::SparseMatrixCSR{0}) where {T}
  rowptr = _copy_and_increment(a.rowptr)
  colval = _copy_and_increment(a.colval)
  Transpose(lu!(translu.parent,SparseMatrixCSC(a.m,a.n,rowptr,colval,a.nzval)))
end
=#

dimlub(_rows) = isempty(_rows) ? 0 : Int(maximum(_rows))
#=
function nnz(x) #sizes, densities)

  for (j, d) in enumerate(densities)
        for (i, n) in enumerate(sizes)
            
            num_nzs = floor(Int, n*n*d)
            rows = rand(1:n, num_nzs)
            cols = rand(1:n, num_nzs)
            vals = rand(num_nzs)
            end 
            end 
           num_nzs
  
  size(x, 1), size(x,2),size(x,3)
end 
=#
# numnz = floor(Int, n*n) size(x, 1), size(x,2)  # for 2d collection
#=
function findnz(sparseMat::SparseMatrixCSR{_rows,_cols,_vals}) where {_rows,_cols,_vals}
  
  numnz =  size(x, 1), size(x,2),size(x,3) #floor(Int, n*n*d)end  size(x, 1), size(x,2) , size(x,3) #floor(Int, n*n) size(x, 1), size(x,2) # nnz(sparseMat)
  _rows = Vector{R}(undef, numnz)
  _cols = Vector{C}(undef, numnz)
  _vals = Vector{Tv}(undef, numnz)
  count = 1
  o = getoffset(S)
  @inbounds for row in 1:size(S,1)
    @inbounds for col in nzrange(S,row)
      _rows[count] = row
      _cols[count] = S.colval[k]+o
      _vals[count] = S.nzval[k]
      count += 1
    end
  end
  return (_rows, _cols, _vals)
end
=#

function transpose()

end 
_precis :: Float64 = 6  # global  #Sneaky issue: `precision` is a julia keyword! # `_precision` is a global var , pick another name ( LoadError: cannot set type for global _precision. It already has a value or is already set to a different type.)
#_precis :: Float64 = 6  #  global

preciseNum = ( 10.0^-(_precis))  # no `float`:  "TypeError: in set_binding_type!, expected Type, got a value of type typeof(float)"
# LoadError: cannot set type for global _precision. It already has a value or is already set to a different type. (rename to `precis` ) 

 _precis = 6.0
x = [1,2,3]
preciseNum = (1/10)^ (_precis*1.0)  #(10.0/1)^-precision*1.0 #float(10)^ float( - precision) #<-- # try 1/x..  or float(x) ^.. OR (x//1)^.. # suggestion didn't work
ans = [input * preciseNum for input in x if input <0]
ans 

function eps(x ;preciseNum=(1/10)^ (_precis*1.0),   _precis::Float64 =6.0 ) #  :: Vector{Float64}
  #preciseNum = (1/10)^ (_precis*1.0)  #(10.0/1)^-precision*1.0 #float(10)^ float( - precision) #<-- # try 1/x..  or float(x) ^.. OR (x//1)^.. # suggestion didn't work
  ans = [input * preciseNum for input in x if input < 0 ]
  ans 
end 

#demo
x = [1,2,3] 
# eps = eps(x)# LoadError: cannot assign a value to variable Base.eps [rename eps]
ep = eps(x)

function abs(x)
	 
  ans = 0
	 n = length(x)
	 
  if n == 1 # scalar
    ans = x
    if x > 0 # do nothing
    elseif x < 0
        ans = -x
    end
	elseif n >1 # collection 
		  ans = [- input for input in x if input <0]
	end # todo: add else
	ans 
end 

# SymSparseMatrixCSR # valid if a 1 single matrix
## import LinearAlgebra: mul!, lu, lu!
#depreciated libraries (view only)
# using sparse
# using base.linAlg,  base.vecdot @depreciated
# LinearAlgebra.ldltfact @depreciated 

# using LinearAlgebra: dot, cross, factorize, cholesky, Diagonal, lowrankdowndate , lowrankupdate!
# Pick an epsilon `\eps` or  `ε`:
# pick a precision ( as precise as: 6, 8 or 10 digits? ) 
power = _precis *1.0
println("power ", typeof(power) )
println("10.0 ", typeof(10.0) )

function Float64(x::Vector{Float64})
   [1.0*input  for input in x if typeof(input) != Float64 ]
end 

ε =  float( 10.0)^-(power)   #1.0* 10^-precision*1.0  #Float64(10)^-6 # that is acceptable (even in forex)  #Note: float64 the base [Fixes: "julia cannot raise integer to negative power" ] #  LoadError: MethodError: no method matching Float64(::Vector{Float64})
a =  ones(11)
ε = abs(Float64(a)) #eps(abs(float(ones(11)))) # pick \eps  #
preciseNum = ε #(1/10)^ (_precis*1.0)  #(10.0/1)^-precision*1.0 #float(10)^ float( - precision) #<-- # try 1/x..  or float(x) ^.. OR (x//1)^.. # suggestion didn't work
input = 1.0
ε  = -1.0*input * preciseNum  #[-1.0*input * preciseNum for input in x if input < 0 ]
 
da= ε*a  #  we have a small step(s) `da`  

# Do not Test of `symmetric` `positive-definite`,  `strided-matrix`
apd  = a'*a

# sparse(A) # ERROR: MethodError: no method matching sparse(::Tuple{Matrix{Float64}, Matrix{Float64}})
# A = SparseMatricesCSR(A) # (A,1)
# function SparseMatrixCSR{Bi} # sparse matix acceptes a sparse BiDirectional matrix 

#= welcome

Matricies are  of two kinds 
1. Dense Matrix  (most exaples: simple ) : factorize, cholesky, cholfact! (efficient)
2. Sparse Matrix  a More General type of Matrixes ( rare: non-simple ) :  # ldlfact (depreciarted (or probably moved) 

=#
#array initialization 
A  = [;;] # init an array
# julia> A  = [;;]
# 0×0 Matrix{Any}


#Dot product

A = ones(10) 
B = ones(10)
dotProduct = dot(A,B)  #A.B # update: not found #mat1 DOT mat2 #[Ensure no spaces are left between matricies]
#
 A = complex.(rand(2, 2), rand(2, 2)) # Hermetian ( Dense, non-spaese by defaust ) # TODOO> help people 
A = SparseMatrixCSR( length(A), length(A), A) # <---

A = (rand(2,2), rand(2,2))
# Possible Issues:
# julia> A = complex . (rand(2,2), rand(2,2)) # we cannot have spaces 
# ERROR: syntax: space before "." not allowed in "complex ." at REPL[87]:1

# julia> A = complex.(rand(2,2), rand(2,2))
# 2×2 Matrix{ComplexF64}:
# 0.691569+0.280538im  0.746668+0.915148im
# 0.332808+0.864212im  0.634443+0.732236im

#Q. Is It positive Definite ? (this needs a sparse matrix ) [Generalized form of a matrix]


"""
```
julia> isposdef(A)
false
```
"""
isposdef(A)

# We can check `A` Array by  `isposdef`(F::Cholesky) = F.info == 0 and 
# Something like isposdef(A::AbstractMatrix) = ishermitian(A) && isposdef(cholfact(A))
# julia> ishermitian(Symmetric(A))
  
#stridex = 1; stridey=2;
#classicDot = dot(10, A, stridex, B, stridey) #nope
#dot(A,B) # mis-match (A is 10, B is 20 ) 
#vectorDot = # vecdot(A,B) # not defined (julia 1
#@test classicDot != vectorDot 
# 
# dot product
dotResult  = A.B

# Cross 
corssed = cross(A, B)

# Factorize
#= Factorize is special: it checks to see whether a certain property is available in Array A
  Some Properties Include: 
1. Positive-definite	Cholesky (use cholfact)
2.Dense Symmetric/Hermitian	Bunch-Kaufman (use bkfact)
3. Sparse Symmetric/Hermitian	LDLt (use ldltfact)
4. Triangular	( use Triangular ) 
5. Diagonal	( use Diagonal ) 
6. Bidiagonal	(use Bidiagonal ) 
7. Tridiagonal	LU (use lufact)
8. Symmetric real tridiagonal	LDLt (use ldltfact)
9. General square	LU (use lufact)
10 General non-square	QR (use qrfact)
=#

# 1. Positive-definite Property
#depreciated list : Base.LinALg, 
isPositiveDefinite = true
# Check whether Array A isPositiveDefinite (via cholesky) 
# PositiveDefinite  = Array(Bidiagonal(A, isPositiveDefinite))

 # Alternative Dorm: call  cholesky's `cholfact`
#Construct a `Cholesky Factorization` of a dense symmetric positive definite matrix
cholesky=  cholfact(A) # Base.LinAlg.Cholesky{Float64,Array{Float64,2}}
@test PositiveDefinite == cholesky(A) # 
# Construct a matrix from the diagonal of A.
Diag = Diagonal(A)  #ok [Julia shows you the acual diagonals in a Matrix]


isSuper = true 

# BiDiagonal 
# is a banded matrix with non-zero entries along the main diagonal and either the diagonal above or the diagonal below. 
# This means there are exactly two non-zero diagonals in the matrix.
#Application 
# Constructs an upper (isupper=true) or lower (isupper=false) bidiagonal matrix using the given diagonal (Diag) 
# and off-diagonal (ev) vectors. 
# the type Bidiagonal and provides efficient `specialized linear solvers`, 
#  but may be converted into a regular matrix with convert(Array, _) (or Array(_) for short). 
#    ev's length must be one less than the length of dv.

# Matrix A can either be a Symmetric or Hermitian StridedMatrix or a perfectly symmetric or Hermitian StridedMatrix. 
# In the latter case, the optional argument `uplo` may be `:L` for using the lower part or `:U` for the upper part of `A`. 
# The default is to use `:U`. The triangular Cholesky factor can be obtained from the factorization `F` with: `F[:L]` and `F[:U]`. 
# The following functions are available for Cholesky
function offdiag2(A) # Mr. Tamas Papp
    [ A[ι] for ι in CartesianIndices(A) if ι[1] ≠ ι[2] ] # list comprehension
end 
# typeof(A)
# Vector{Float64} (alias for Array{Float64, 1})

Offdiagonal = offdiag2(A)

# Bidiagonal 

# use :L for using the lower part or :U for the upper part of A
    
A = [4. 12. -16.; 12. 37. -43.; -16. -43. 98.]
# BidiagonalLo, BidiagonalUp = cholesky[:L]  cholesky[:U]
print("Lower bidiagonal =",BidiagLo , "\nUpper bidiagonal = ", bidiagUp ,"\n")
 A == cholesky[:L] * cholesky[:U]

#Note: `cholfact!` not defined 
#cholfact!(A) #same as cholesky by saves space by overwritting  the input A, instead of creating a copy [  (A, [uplo::Symbol,] Val{false}) -> Cholesky ] 
print(cholfact!(A) == cholesky ) 
# Warning: InexactError exception is thrown if the factorization produces a number not representable by the element type of A, e.g. for integer types.

#lowrankdowndate! (C::Cholesky, v::StridedVector) -> CC::Cholesky
# Update a Cholesky factorization C with the vector v

lowrankdowndate 
# If A = C[:U]'C[:U] then CC = cholfact(C[:U]'C[:U] + v*v') # CC only uses O(n^2) 
# If A = C[:U]'C[:U] then CC = cholfact(C[:U]'C[:U] - v*v')
A = C[:U]'C[:U]
CC = cholfact( C[:U]'C[:U] - v*v')
print(A == CC)
#C is updated in place such that on exit `C == CC`. However, The vector v is destroyed during the computation

#Note: A fill-reducing permutation is used. `F = ldltfact(A)` is most frequently used to solve systems of equations `A*x = `b with `F\b`
# Test cholesky with Symmetric/Hermitian upper/lower

symmetricUpper  = Symmetric(apd) # Upper Symmetric 
symmetricLower = Symmetric(apd, :L) # Lower Symmetric
hermitianUpper  = Hermitian(apd) # Upper hermitian
hermitianLower = Hermitian(apd, :L) # Lower Hermitian
# ErrorMatrix = Expected - 
E = abs.(apd - Matrix(capd))

for i=1:n, j=1:n
  @test E[i,j] <= (n+1)ε/(1-(n+1)ε)*real(sqrt(apd[i,i]*apd[j,j]))
end

capd  = factorize(apd)
  
#note: A can either  a Symmetric or Hermitian == StridedMatrix)
#@test 

end 
