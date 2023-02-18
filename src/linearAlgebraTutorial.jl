module linearAlgebraTutorial

using LinearAlgebra: dot, cross, factorize, cholesky, Diagonal , cholfact! , lowrankdowndate  , lowrankupdate!
using Test, LinearAlgebra, Random
using LinearAlgebra: issymmetric, isposdef # is positive definite
#using SparseMatrixDicts: sparse #works  but maybe not for dedicated matrices 
using SparseMatricesCSR: SparseMatricesCSR
# SymSparseMatrixCSR # valid if a 1 single matrix
## import LinearAlgebra: mul!, lu, lu!
#depreciated libraries (view only)
# using sparse
# using base.linAlg,  base.vecdot @depreciated
# LinearAlgebra.ldltfact @depreciated 

# using LinearAlgebra: dot, cross, factorize, cholesky, Diagonal, lowrankdowndate , lowrankupdate!
# Pick an epsilon `\eps` or  `ε`:
# pick a precision ( as precise as: 6, 8 or 10 digits? ) 

ε = 10^-6 # that is acceptable (even in forex) 
a =  ones(11)
ε = = eps(abs(float(one(eltya)))) # pick \eps 
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
B = ones(20)
#
 A = complex.(rand(2, 2), rand(2, 2)) # Hermetian ( Dense, non-spaese by defaust ) # TODOO> help people 
A = SparseMatricesCSR( length(A), length(A), A)

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

# use :L for using the lower part or :U for the upper part of A
    
A = [4. 12. -16.; 12. 37. -43.; -16. -43. 98.]
BidiagonalLo, BidiagonalUp = cholesky[:L]  cholesky[:U]
print("Lower bidiagonal =",BidiagLo , "\nUpper bidiagonal = ", bidiagUp ,"\n")
 A == cholesky[:L] * cholesky[:U]

cholfact!(A) #same as cholesky by saves space by overwritting  the input A, instead of creating a copy [  (A, [uplo::Symbol,] Val{false}) -> Cholesky ] 
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
