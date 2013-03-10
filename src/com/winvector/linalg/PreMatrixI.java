package com.winvector.linalg;

import java.io.Serializable;
import java.util.ArrayList;

import com.winvector.linalg.sparse.SparseVec;


public interface PreMatrixI extends LinOpI,Serializable {
	SparseVec extractColumn(int ci);
	PreMatrixI addColumns(final ArrayList<SparseVec> cs); 
	PreMatrixI extractColumns(int[] basis);
	PreMatrixI extractRows(final int[] rb);
	
	double[] sumAbsRowValues();
	PreMatrixI rescaleRows(double[] scale);
	
	int[] colBasis(final int[] forcedCols, final double minVal);
	
	/**
	 * not a preferred method (can be slow)
	 * @param i
	 * @param j
	 * @return
	 */
	double get(int i, int j);
}
