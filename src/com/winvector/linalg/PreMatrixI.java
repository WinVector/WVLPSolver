package com.winvector.linalg;

import java.io.Serializable;

import com.winvector.linalg.sparse.SparseVec;


public interface PreMatrixI extends LinOpI,Serializable {
	SparseVec extractColumn(int ci);
	PreMatrixI extractColumns(int[] basis);
	
	/**
	 * not a preferred method (can be slow)
	 * @param i
	 * @param j
	 * @return
	 */
	double get(int i, int j);
}
