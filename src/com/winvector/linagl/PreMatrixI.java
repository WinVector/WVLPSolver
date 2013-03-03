package com.winvector.linagl;

import java.io.Serializable;


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
