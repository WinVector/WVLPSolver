package com.winvector.linalg;

import java.io.Serializable;

import com.winvector.linalg.sparse.SparseVec;


public interface PreMatrixI extends LinOpI,Serializable {
	Object buildExtractTemps();
	SparseVec extractColumn(int ci, Object extractTemps);
	PreMatrixI extractColumns(int[] basis);
	<T extends Matrix<T>> T matrixCopy(LinalgFactory<T> factory);
	
	/**
	 * not a preferred method (can be slow)
	 * @param i
	 * @param j
	 * @return
	 */
	double get(int i, int j);
}
