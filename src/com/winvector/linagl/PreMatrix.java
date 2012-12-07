package com.winvector.linagl;

import java.io.Serializable;

public interface PreMatrix extends Serializable {
	int rows();
	int cols();
	double[] multLeft(double[] y);
	double[] mult(double[] x);
	<T extends Matrix<T>> T extractColumns(int[] basis, LinalgFactory<T> factory);
	<T extends Matrix<T>> T matrixCopy(LinalgFactory<T> factory);
}
