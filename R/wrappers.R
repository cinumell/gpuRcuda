#' @useDynLib gpuRcuda
#' @importFrom Rcpp evalCpp
#' @import methods

# Matrix Multiplication
gpu_Mat_mult <- function(A, B){
	
	type <- typeof(A)
	
	C <- cudaMatrix(nrow=nrow(A), ncol=ncol(B), type=type)
	
	switch(type,
				 integer = {stop("integer not currently implemented")},
				 float = {cpp_vienna_cudaMatrix_sgemm(A@address,
				 																		B@address,
				 																		C@address)
				 },
				 double = {
				 	if(!deviceHasDouble()){
				 		stop("Selected GPU does not support double precision")
				 	}else{cpp_vienna_cudaMatrix_dgemm(A@address,
				 																	 B@address,
				 																	 C@address)
				 	}
				 },
				{
					stop("type not recognized")
				})
	
	return(C)
}
