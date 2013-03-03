package com.winvector.linagl;

public interface LinOpI {
	int rows();
	int cols();
	double[] multLeft(double[] y);
	double[] mult(double[] x);
}
