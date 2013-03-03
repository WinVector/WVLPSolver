package com.winvector.lp.impl;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import com.winvector.linagl.LinalgFactory;
import com.winvector.linagl.Matrix;
import com.winvector.linagl.SparseVec;
import com.winvector.lp.AbstractLPEQProb;
import com.winvector.lp.LPException;
import com.winvector.lp.LPException.LPErrorException;

/**
 * Basis enhanced with extra record keeping
 * @author johnmount
 *
 */
final class EnhancedBasis<T extends Matrix<T>> implements Serializable {
	private static final long serialVersionUID = 1L;

	public final AbstractLPEQProb prob;

	public final int m;  // rank of basis
	public final int[] basis; // variables in basis
	public final Set<Integer> curBasisSet = new HashSet<Integer>(); // set of variables in basis
	
	public final LinalgFactory<T> factory;
	private final int[] binvNZJTmp;
	public T BInv = null;
	// run counters
	private int normalSteps = 0;


	/**
	 * try to use inverse to solve (if present)
	 * 
	 * @param y
	 * @return x s.t. x = BInv y (if BInv!=null) 
	 *         prob.extractColumns(basis) x = y (otherwise)
	 *         (want x>=0)
	 * @throws LPErrorException 
	 */
	public double[] basisSolveRight(final double[] y) throws LPErrorException {
		if(null==BInv) {
			try {
				BInv = prob.extractColumns(basis,factory).inverse();
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
	 * @return x s.t. x = BInv y (if BInv!=null) 
	 *         prob.extractColumns(basis) x = y (otherwise)
	 *         (want x>=0)
	 * @throws LPErrorException 
	 */
	public double[] basisSolveRight(final SparseVec y) throws LPErrorException {
		if(null==BInv) {
			try {
				BInv = prob.extractColumns(basis,factory).inverse();
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
	 *         prob.extractColumns(basis) = y (otherwise)
	 * @throws LPErrorException 
	 */
	public double[] basisSolveLeft(final double[] y) throws LPErrorException {
		if(null==BInv) {
			try {
				BInv = prob.extractColumns(basis,factory).inverse();
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
	 * @param basisColumns
	 *            m-vector that is a valid starting basis
	 */
	public EnhancedBasis(final AbstractLPEQProb prob_in, final int[] basis_in, final LinalgFactory<T> factory) throws LPException {
		this.factory = factory;
		prob = prob_in;
		//RevisedSimplexSolver.checkParams(prob.A, prob.b, prob.c, basis_in);
		m = prob.rows();
		binvNZJTmp = new int[m];
		basis = new int[basis_in.length];
		for (int i = 0; i < basis.length; ++i) {
			basis[i] = basis_in[i];
			curBasisSet.add(basis[i]);
		}
		try {
			BInv = prob.extractColumns(basis,factory).inverse();
		} catch (Exception e) {
			throw new LPErrorException("couldn't invert initial basis");
		}
	}
	

	public double[] leftBasisSoln() throws LPErrorException {
		final double[] cB = new double[m];
		for(int i=0;i<m;++i) {
			cB[i] = prob.c(basis[i]);
		}
		final double[] lambda = basisSolveLeft(cB);
		return lambda;
	}
	
	public double computeRI(final double[] lambda, final int vi) {
		final double cFi = prob.c(vi);
		final double lambdaFi = prob.extractColumn(vi).dot(lambda);
		final double ri = cFi - lambdaFi; 
		return ri;
	}
	


	public void basisPivot(final int leavingI, final int enteringV, final double[] binvu) throws LPErrorException {
		curBasisSet.remove(basis[leavingI]);
		curBasisSet.add(enteringV);
		basis[leavingI] = enteringV;
		++normalSteps;
		if(normalSteps%(25*m+1)==0) {
			BInv = null; // forced refresh
			// ideas is BInv is getting unreliable due to rounding
			// a refresh takes around O(m^3) steps and updates take O(m^2) steps.
			// so every m steps we can hide the extra m^3 work which amortizes to m^3/m per-step 
			// of a refresh
		}
		if (BInv != null) {
			// rank 1 update the inverse
			final double vKInv = 1.0/binvu[leavingI];
			int nextJJ = 0;
			for(int j=0;j<m;++j) {
				if(Math.abs(BInv.get(leavingI,j))>1.0e-8) {
					binvNZJTmp[nextJJ] = j;
					++nextJJ;
				}
			}
			final int nJJ = nextJJ;
			for(int i=0;i<m;++i) {
				if(leavingI!=i) {
					final double binvui = binvu[i];
					if(binvui!=0.0) {
						final double vi = -binvui*vKInv;
						for(nextJJ=0;nextJJ<nJJ;++nextJJ) {
							final int j = binvNZJTmp[nextJJ];
							BInv.set(i, j,BInv.get(i, j)+vi*BInv.get(leavingI,j));
						}
					}
				}
			}
			for(nextJJ=0;nextJJ<nJJ;++nextJJ) {
				final int j = binvNZJTmp[nextJJ];
				BInv.set(leavingI, j,vKInv*BInv.get(leavingI,j));
			}
		} else {
			try {
				BInv = prob.extractColumns(basis,factory).inverse();
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
		curBasisSet.clear();
		for(int i=0;i<basis.length;++i) {
			basis[i] = d[i];
			curBasisSet.add(basis[i]);
		}
		try {
			BInv = prob.extractColumns(basis,factory).inverse();
		} catch (Exception e) {
			throw new LPErrorException("couldn't invert reset basis");
		}
	}
}
