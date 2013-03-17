package com.winvector.lp.impl;

import java.io.Serializable;

import com.winvector.linalg.LinalgFactory;
import com.winvector.linalg.Matrix;
import com.winvector.linalg.sparse.SparseVec;
import com.winvector.linalg.sparse.TabularLinOp;
import com.winvector.lp.LPEQProbI;
import com.winvector.lp.LPException;
import com.winvector.lp.LPException.LPErrorException;

/**
 * Basis enhanced with extra record keeping
 * @author johnmount
 *
 */
final class EnhancedBasis<T extends Matrix<T>> implements Serializable {
	private static final long serialVersionUID = 1L;
	private final double epsilon = 1.0e-8; // zeroness test in inverse


	public final LPEQProbI prob;

	public final int m;  // rank of basis
	public final int[] basis; // variables in basis
	
	private final LinalgFactory<T> factory;
	private final int[] binvNZJTmp;
	T binvW = null;
	private final TabularLinOp binvS;
	private final double[] cBTemp;
	// run counters
	private long normalSteps = 0;


	private void readyBinv() throws LPErrorException {
		if(null==binvW) {
			try {
				binvW = factory.matrixCopy(prob.extractColumns(basis)).inverse();
				if(null!=binvS) {
					binvS.setV(binvW);
				}
			} catch (Exception e) {
				throw new LPErrorException("couldn't invert basis");
			}
		}
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
	public double[] basisSolveRight(final double[] y) throws LPErrorException {
		if((null!=binvS)&&(binvS.valid())) {
			return binvS.mult(y);
		} else {
			return binvW.mult(y);
		}
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
		if((null!=binvS)&&(binvS.valid())) {
			return binvS.mult(y);
		} else {
			return binvW.mult(y);
		}
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
		if((null!=binvS)&&(binvS.valid())) {
			return binvS.multLeft(y);
		} else {
			return binvW.multLeft(y);
		}
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
	public EnhancedBasis(final LPEQProbI prob_in, final int[] basis_in, final LinalgFactory<T> factory) throws LPException {
		this.factory = factory;
		prob = prob_in;
		m = prob.rows();
		cBTemp = new double[m];
		//RevisedSimplexSolver.checkParams(prob.A, prob.b, prob.c, basis_in);
		binvNZJTmp = new int[m];
		basis = new int[basis_in.length];
		for (int i = 0; i < basis.length; ++i) {
			basis[i] = basis_in[i];
		}
		binvS = null; // new TabularLinOp(m,m,10000);
		readyBinv();
	}
	

	double[] leftBasisSoln() throws LPErrorException {
		for(int i=0;i<m;++i) {
			cBTemp[i] = prob.c(basis[i]);
		}
		final double[] lambda = basisSolveLeft(cBTemp);
		return lambda;
	}
	
	public double computeRI(final double[] lambda, final int vi) {
		final double cFi = prob.c(vi);
		final double lambdaFi = prob.extractColumn(vi).dot(lambda);
		final double ri = cFi - lambdaFi; 
		return ri;
	}
	


	public void basisPivot(final int leavingI, final int enteringV, final double[] binvu) throws LPErrorException {
		basis[leavingI] = enteringV;
		++normalSteps;
		if(normalSteps%(25*m+1)==0) {
			binvW = null; // forced refresh
			binvS.invalidate();
			// ideas is BInv is getting unreliable due to rounding
			// a refresh takes around O(m^3) steps and updates take O(m^2) steps.
			// so every m steps we can hide the extra m^3 work which amortizes to m^3/m per-step 
			// of a refresh
		}
		if (binvW != null) {
			// rank 1 update the inverse
			final double vKInv = 1.0/binvu[leavingI];
			int nextJJ = 0;
			for(int j=0;j<m;++j) {
				if(Math.abs(binvW.get(leavingI,j))>epsilon) {
					binvNZJTmp[nextJJ] = j;
					++nextJJ;
				}
			}
			final int nJJ = nextJJ;
			for(int i=0;i<m;++i) {
				if(leavingI!=i) {
					final double binvui = binvu[i];
					if(Math.abs(binvui)>epsilon) {
						final double vi = -binvui*vKInv;
						for(nextJJ=0;nextJJ<nJJ;++nextJJ) {
							final int j = binvNZJTmp[nextJJ];
							binvW.set(i, j,binvW.get(i, j)+vi*binvW.get(leavingI,j));
						}
					}
				}
			}
			for(nextJJ=0;nextJJ<nJJ;++nextJJ) {
				final int j = binvNZJTmp[nextJJ];
				binvW.set(leavingI, j,vKInv*binvW.get(leavingI,j));
			}
			if(null!=binvS) {
				binvS.setV(binvW);
			}
		} else {
			readyBinv();
		}
	}

	public void resetBasis(final int[] d) throws LPErrorException {
		for(int i=0;i<basis.length;++i) {
			basis[i] = d[i];
		}
		binvW = null;
		readyBinv();
	}
}
