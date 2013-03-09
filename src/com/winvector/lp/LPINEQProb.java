package com.winvector.lp;

import java.util.ArrayList;

import com.winvector.linalg.DenseVec;
import com.winvector.linalg.PreMatrixI;
import com.winvector.linalg.PreVecI;
import com.winvector.linalg.sparse.ColumnMatrix;
import com.winvector.linalg.sparse.SparseVec;
import com.winvector.lp.LPException.LPMalformedException;

/**
 * primal: A x <= b, x>=0 minimize c.x
 * dual: y A <= c, y >=0 max y.b
 * @author johnmount
 *
 */
public final class LPINEQProb extends LPProbBase {
	private static final long serialVersionUID = 1L;

	public LPINEQProb(final PreMatrixI A_in, final double[] b_in, final PreVecI c_in)
			throws LPException.LPMalformedException {
		super(A_in,b_in,c_in,"<=");
	}
	
	@Override
	public LPEQProb eqForm() throws LPMalformedException {
		final int m = A.rows();
		final int n = A.cols();
		final ArrayList<SparseVec> slacks = new ArrayList<SparseVec>(m);
		for(int i=0;i<m;++i) {
			slacks.add(SparseVec.sparseVec(m,i,1.0));
		}
		final double[] newc = new double[n+m];
		for(int j=0;j<n;++j) {
			newc[j] = c.get(j);
		}
		return new LPEQProb(new ColumnMatrix(A).addColumns(slacks),b,new DenseVec(newc));
	}
}
