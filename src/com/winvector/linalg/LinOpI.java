package com.winvector.linalg;

import com.winvector.linalg.sparse.HVec;

public interface LinOpI {
	int rows();
	int cols();
	double[] multLeft(double[] y);
	double[] mult(double[] x);
	double[] mult(HVec x);
}
