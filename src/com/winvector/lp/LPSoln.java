package com.winvector.lp;

import java.io.Serializable;

import com.winvector.linalg.sparse.HVec;




public final class LPSoln implements Serializable {
	private static final long serialVersionUID = 1L;

	public final HVec primalSolution;
	public final int[] basisColumns;
	public long reportedRunTimeMS;
	public int[] basisRows;

	public LPSoln(final HVec primalSoln_in, final int[] basisColumns_in, final int[] basisRows_in,
			final long reportedRunTimeMS_in) {
		primalSolution = primalSoln_in;
		basisColumns = basisColumns_in;
		basisRows = basisRows_in;
		reportedRunTimeMS = reportedRunTimeMS_in;
	}

	
	@Override
	public String toString() {
		final StringBuilder b = new StringBuilder();
		if (primalSolution != null) {
			b.append(primalSolution);
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