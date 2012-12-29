package com.winvector.lp;

import java.io.Serializable;

import com.winvector.linagl.Matrix;


public final class LPSoln implements Serializable {
	private static final long serialVersionUID = 1L;

	public final double[] primalSolution;
	public final int[] basisColumns;
	public int[] basisRows;
	public int stepsTaken = 0;

	public LPSoln(final double[] primalSoln_in, final int[] basisColumns_in, final int[] basisRows_in) {
		primalSolution = primalSoln_in;
		basisColumns = basisColumns_in;
		basisRows = basisRows_in;
	}
	
	// TODO: replace this constructor
	public LPSoln(final double[] primalSoln_in, final int[] basisColumns_in) {
		this(primalSoln_in,basisColumns_in,null);
	}

	public String toString() {
		final StringBuilder b = new StringBuilder();
		if (primalSolution != null) {
			b.append(Matrix.toString(primalSolution));
		}
		if (basisColumns != null) {
			b.append(" (");
			for (int i = 0; i < basisColumns.length; ++i) {
				if (i > 0) {
					b.append(" ");
				}
				b.append(basisColumns[i]);
			}
			b.append(")");
		}
		return b.toString();
	}
}