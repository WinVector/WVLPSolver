package com.winvector.lp;

import java.util.Random;

import com.winvector.linalg.LinalgFactory;
import com.winvector.linalg.Matrix;
import com.winvector.linalg.sparse.SparseVec;

/**
 * represents primal: min c.x: A x = b, x>=0
 * @author johnmount
 *
 */
public interface LPEQProbI {
	int rows();
	double[] b();
	double c(int i);
	Object buildExtractTemps();
	SparseVec extractColumn(int j, Object extractTemps);
	<T extends Matrix<T>> T extractColumns(int[] basis, LinalgFactory<T> factory);
	InspectionOrder buildOrderTracker(Random rand);
}
