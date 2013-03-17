package com.winvector.lp;

import java.util.Random;

import com.winvector.linalg.PreMatrixI;
import com.winvector.linalg.sparse.SparseVec;

/**
 * represents 
 *  primal: min c.x: A x = b, x>=0
 *  dual: max y.b: y A <= c (no sign conditions on y)
 * @author johnmount
 *
 */
public interface LPEQProbI {
	int rows();
	double[] b();
	double c(int i);
	SparseVec extractColumn(int j);
	PreMatrixI extractColumns(int[] basis);
	InspectionOrder buildOrderTracker(Random rand);
}
