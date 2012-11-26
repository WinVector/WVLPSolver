package com.winvector.lp.impl;

import java.io.Serializable;
import java.util.Arrays;

import com.winvector.linagl.Matrix;
import com.winvector.lp.LPException;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPException.LPErrorException;

/**
 * not a traditional Tableau as we are not implementing the row operations that update the Tableau.
 * We instead are abstracting that down to an inverse linear operator that we incrementally update.
 * This is slower but clearer (you need to theory of actual tableau bookkeeping as you have reduced to linear algebra)
 * @author johnmount
 *
 */
final class RTableau<Z extends Matrix<Z>> implements Serializable {
	private static final long serialVersionUID = 1L;

	public final LPEQProb<Z> prob;

	public final int m;

	public final int n;

	public final int[] basis;
	
	public Z BInv = null;

	public int normalSteps = 0;


	/**
	 * try to use inverse to solve (if present)
	 * 
	 * @param y
	 * @return x s.t. x = BInv y (if BInv!=null) 
	 *         prob.A.extractColumns(basis) x = y (otherwise)
	 *         (want x>=0)
	 * @throws LPErrorException 
	 */
	public double[] basisSolveRight(final double[] y) throws LPErrorException {
		if(null==BInv) {
			try {
				BInv = prob.A.extractColumns(basis).inverse();
			} catch (Exception e) {
				throw new LPErrorException("couldn't invert basis");
			}
		}
		return BInv.mult(y);
	}

	/**
	 * try to use inverse to solve (if present)
	 * 
	 * @param y
	 * @return x s.t. x = y BInv (if BInv!=null) or x
	 *         prob.A.extractColumns(basis) = y (otherwise)
	 * @throws LPErrorException 
	 */
	public double[] basisSolveLeft(final double[] y) throws LPErrorException {
		if(null==BInv) {
			try {
				BInv = prob.A.extractColumns(basis).inverse();
			} catch (Exception e) {
				throw new LPErrorException("couldn't invert basis");
			}
		}
		return BInv.multLeft(y);
	}

	/**
	 * build initial RTableu from Gilbert String Linear Algebra and its
	 * Applications second edition. Section 8.2 The Simplex Method (pp.
	 * 320-321)
	 * 
	 * @param prob
	 *            well formed LPProb
	 * @param basis
	 *            m-vector that is a valid starting basis
	 */
	public RTableau(final LPEQProb<Z> prob_in, final int[] basis_in) throws LPException {
		prob = prob_in;
		//RevisedSimplexSolver.checkParams(prob.A, prob.b, prob.c, basis_in);
		m = prob.A.rows();
		n = prob.A.cols();
		basis = new int[basis_in.length];
		for (int i = 0; i < basis.length; ++i) {
			basis[i] = basis_in[i];
		}
		try {
			BInv = prob.A.extractColumns(basis).inverse();
		} catch (Exception e) {
			throw new LPErrorException("couldn't invert initial basis");
		}
	}
	

	public double[] leftBasisSoln(final double[] c) throws LPErrorException {
		final double[] cB = Matrix.extract(c,basis);
		final double[] lambda = basisSolveLeft(cB);
		return lambda;
	}
	
	public double computeRI(final double[] lambda, final double[] c, final int vi) throws LPErrorException {
		final double cFi = c[vi];
		final double[] Fi = prob.A.extractColumn(vi);
		final double lambdaFi = Matrix.dot(Fi,lambda);
		final double ri = cFi - lambdaFi; 
		return ri;
	}

	public void basisPivot(final int leavingI, final int enteringV, final double[] v) throws LPErrorException {
		basis[leavingI] = enteringV;
		++normalSteps;
		if(normalSteps%(m+n+1)==0) {
			BInv = null; // forced refresh
		}
		if (BInv != null) {
			// rank 1 update the inverse
			final double vKInv = 1.0/v[leavingI];
			for(int i=0;i<m;++i) {
				if(leavingI!=i) {
					final double vi = -v[i]*vKInv;
					for(int j=0;j<m;++j) {
						BInv.set(i, j,BInv.get(i, j)+vi*BInv.get(leavingI,j));
					}
				}
			}
			for(int j=0;j<m;++j) {
				BInv.set(leavingI, j,vKInv*BInv.get(leavingI,j));
			}
		} else {
			try {
				BInv = prob.A.extractColumns(basis).inverse();
			} catch (Exception e) {
				throw new LPErrorException("couldn't invert intermediate basis");
			}
		}
	}

	public int[] basis() {
		final int[] r = new int[m];
		for (int i = 0; i < m; ++i) {
			r[i] = basis[i];
		}
		if (r.length > 1) {
			Arrays.sort(r);
		}
		return r;
	}

	public void resetBasis(final int[] d) throws LPErrorException {
		for(int i=0;i<basis.length;++i) {
			basis[i] = d[i];
		}
		try {
			BInv = prob.A.extractColumns(basis).inverse();
		} catch (Exception e) {
			throw new LPErrorException("couldn't invert reset basis");
		}
	}
}
