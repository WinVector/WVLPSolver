package com.winvector.linagl;

public class DenseVec implements PreVecI {
	private final double[] x;
	
	public DenseVec(final double[] x) {
		this.x = x;
	}

	public DenseVec(final PreVecI c) {
		final int dim = c.dim();
		x = new double[dim];
		for(int i=0;i<dim;++i) {
			x[i] = c.get(i);
		}
	}

	@Override
	public int dim() {
		return x.length;
	}

	@Override
	public double get(final int i) {
		return x[i];
	}
	
	public void set(final int i, final double v) {
		x[i] = v;
	}

}
