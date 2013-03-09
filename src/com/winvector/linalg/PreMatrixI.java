package com.winvector.linalg;

import java.io.Serializable;

import com.winvector.linalg.sparse.SparseVec;


public interface PreMatrixI extends LinOpI,Serializable {
	Object buildExtractTemps();
	SparseVec extractColumn(int ci, Object extractTemps);
	<T extends Matrix<T>> T extractColumns(int[] basis, LinalgFactory<T> factory);
	<T extends Matrix<T>> T matrixCopy(LinalgFactory<T> factory);
	/**
	 * not primary method
	 * @param i
	 * @param j
	 * @return
	 */
	double get(int i, int j);
}
