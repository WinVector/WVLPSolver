package com.winvector.lp;

import java.io.Serializable;

import com.winvector.linagl.Matrix;


public final class LPSoln implements Serializable {
	private static final long serialVersionUID = 1L;

	public final double[] x;
	public final int[] basis;

	public LPSoln(final double[] x_in, final int[] basis_in) {
		x = x_in;
		basis = basis_in;
	}

	public String toString() {
		final StringBuilder b = new StringBuilder();
		if (x != null) {
			b.append(Matrix.toString(x));
		}
		if (basis != null) {
			b.append(" (");
			for (int i = 0; i < basis.length; ++i) {
				if (i > 0) {
					b.append(" ");
				}
				b.append(basis[i]);
			}
			b.append(")");
		}
		return b.toString();
	}
}