package com.winvector.lp.impl;

import java.util.ArrayList;
import java.util.Arrays;

import com.winvector.linalg.DenseVec;
import com.winvector.linalg.LinalgFactory;
import com.winvector.linalg.Matrix;
import com.winvector.linalg.PreMatrixI;
import com.winvector.linalg.PreVecI;
import com.winvector.linalg.sparse.HVec;
import com.winvector.linalg.sparse.SparseVec;
import com.winvector.lp.EarlyExitCondition;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPEQProbI;
import com.winvector.lp.LPException;
import com.winvector.lp.LPSoln;
import com.winvector.lp.LPSolver;

/**
 * primal: min c.x: A x = b, x>=0 dual: max y.b: y A <= c y b = y A x <= c x (by
 * y A <=c, x>=0) , so y . b <= c . x at optimial y.b = c.x
 * 
 * only need to directly implement: solve min c.x: A x = b, x>=0, m>n A full row
 * rank, given basis-0 m-vector ( A(basis0) = square matrix of basis0 columns
 * x(basis0) = vector with entries selected by basis0 then x(basis0) =
 * A(basis0)^-1 b, x>=0 and x=0 for non-basis elements)
 */
abstract class LPSolverImpl implements LPSolver {
	public int verbose = 0;
	public double minBasisEpsilon = 1.0e-5;
	public boolean rescale = false;

	/**
	 * from Gilbert String Linear Algebra and its Applications second edition.
	 * Section 8.2 The Simplex Method (pp. 316..323)
	 * 
	 * @param A
	 *            matrix m-row by n-column matrix- full row rank, m <=n
	 * @param b
	 *            m-vector
	 * @param c
	 *            n-vector
	 * @param basis
	 *            m-vector that is a valid starting basis
	 * @throws LPException.LPMalformedException
	 *             if parameters don't match defs
	 */
	public static void checkParams(final PreMatrixI A, final double[] b, final PreVecI c, final int[] basis)
			throws LPException.LPMalformedException {
		if ((A == null) || (b == null) || (c == null) || (basis == null)
				|| (A.rows() <= 0) || (A.rows() != b.length)
				|| (A.rows() != basis.length) || (A.cols() != c.dim())) {
			String problem = "misformed problem";
			if (A == null) {
				problem = problem + " A==null";
			} else {
				if (A.rows() <= 0) {
					problem = problem + " A.rows()<=0";
				}
			}
			if (b == null) {
				problem = problem + " b==null";
			} else {
				if (A != null) {
					if (A.rows() != b.length) {
						problem = problem + " A.rows()(" + A.rows()
								+ ")!=b.length(" + b.length + ")";
					}
				}
			}
			if (c == null) {
				problem = problem + " c==null";
			} else {
				if ((A != null) && (A.rows() > 0)) {
					if (A.cols() != c.dim()) {
						problem = problem + " A.cols()(" + A.cols()
								+ ")!=c.length(" + c.dim() + ")";
					}
				}
			}
			if (basis == null) {
				problem = problem + " basis==null";
			} else {
				if (A != null) {
					if (A.rows() != basis.length) {
						problem = problem + " A.rows()(" + A.rows()
								+ ")!=basis.length(" + basis.length + ")";
					}
				}
			}
			throw new LPException.LPMalformedException(problem);
		}
		int m = A.rows();
		int n = A.cols();
		if (m > n) {
			throw new LPException.LPMalformedException("m>n");
		}
		for (int i = 0; i < basis.length; ++i) {
			if ((basis[i] < 0) || (basis[i] >= n)) {
				throw new LPException.LPMalformedException(
						"out of range column in basis");
			}
		}
		final int[] bSorted = basis.clone();
		Arrays.sort(bSorted);
		for (int i = 1; i < basis.length; ++i) {
			if(bSorted[i-1]==bSorted[i]) {
				throw new LPException.LPMalformedException(
						"duplicate column in basis");
			}
		}
	}


	static String stringBasis(final int[] b) {
		if (b == null) {
			return null;
		}
		StringBuilder r = new StringBuilder();
		r.append('[');
		for (int i = 0; i < b.length; ++i) {
			if (i > 0) {
				r.append(' ');
			}
			r.append(b[i]);
		}
		r.append(']');
		return r.toString();
	}

	/**
	 * find: min c.x: A x = b, x>=0
	 * 
	 * @param prob
	 *            valid LPProb with full row rank
	 * @param basis0
	 *            m-vector that is a valid starting basis
	 * @param l
	 *            (optional) lower bound on desired solution. ( A(basis0) =
	 *            square matrix of basis0 columns x(basis0) = vector with
	 *            entries selected by basis0 then x(basis0) = A(basis0)^-1 b,
	 *            x>=0 and x=0 for non-basis elements) sorted basis0[i+1] >
	 *            basis0[i]
	 * @return optimal basis (need not be sorted)
	 * @throws LPException
	 *             (if infeas or unbounded) no need to check feasibility of
	 *             input or output (check by wrapper)
	 */
	protected abstract <T extends Matrix<T>> LPSoln rawSolve(LPEQProbI prob, int[] basis0,
			final double tol, final int maxRounds, final LinalgFactory<T> factory, final EarlyExitCondition earlyExitCondition) throws LPException;

	/**
	 * @param A
	 *            matrix m-row by n-column matrix- full row rank
	 * @param b
	 *            m-vector
	 * @return basis0 m-vector that is a valid starting basis ( A(basis0) =
	 *         square matrix of basis0 columns x(basis0) = vector with entries
	 *         selected by basis0 then x(basis0) = A(basis0)^-1 b, x>=0 and x=0
	 *         for non-basis elements)
	 * @throws LPException
	 *             (if infeas or unbounded)
	 * 
	 * phase 1 get a basis 
	 */
	private <T extends Matrix<T>> int[] solvePhase1(final PreMatrixI A, final double[] b, final PreVecI cin, final double tol, 
			final int maxRounds, final LinalgFactory<T> factory) 
			throws LPException {
		final int m = A.rows();
		final int n = A.cols();
		final int[] basis0 = new int[m];
		final ArrayList<SparseVec> artificialSlackCols = new ArrayList<SparseVec>(m);
		{ // find if any columns we have are already usable in place of slacks
			Arrays.fill(basis0,-1);
			for(int j=0;j<n;++j) {
				final SparseVec col = A.extractColumn(j);
				if(col.popCount()==1) {
					final int i = col.nzIndex();
					final double vi = col.get(i);
					if(basis0[i]<0) {
						if((b[i]==0)||((b[i]>=0)==(vi>=0))) {
							basis0[i] = j;
						}
					}
				}
			}
			for(int i=0;i<m;++i) {
				if(basis0[i]<0) {
					basis0[i] = n+artificialSlackCols.size();
					artificialSlackCols.add(SparseVec.sparseVec(m,i,b[i]>=0?1.0:-1.0));
				}
			}
			Arrays.sort(basis0);
			if(artificialSlackCols.isEmpty()) {
				return basis0;
			}
		}
		final PreMatrixI AP = A.addColumns(artificialSlackCols);
		final double[] c = new double[n + artificialSlackCols.size()];
		if(null!=cin) {
			// hint at actual objective fn
			double sumAbscin = 0.0;
			for(int i=0;i<n;++i) {
				sumAbscin += Math.abs(cin.get(i));
			}
			final double scale = 1.0e-7/(1.0+sumAbscin);
			for(int i=0;i<n;++i) {
				c[i] = scale*cin.get(i);
			}
		}
		for(int i=n;i<n+artificialSlackCols.size();++i) {
			c[i] = 1.0;
		}
		final LPEQProb p1prob = new LPEQProb(AP, b, new DenseVec(c));
		LPSoln soln = rawSolve(p1prob, basis0, tol, maxRounds, factory, new EarlyExitCondition() {
			@Override
			public boolean canExit(final int[] basis) {
				for(final int bi: basis) {
					if(bi>=n) {
						return false;
					}
				}
				return true;
			}
		});
		if ((soln == null) || (soln.basisColumns == null)
				|| (soln.basisColumns.length != basis0.length) || (soln.primalSolution == null)) {
			throw new LPException.LPErrorException(
					"bad basis back from phase1 raw solve");
		}
		// check objective value is zero
		final double v = soln.primalSolution.dot(c);
		if (Math.abs(v)>1.0e-5) {
			throw new LPException.LPInfeasibleException("primal infeasible");
		}
		// check basis is good
		if (soln.basisColumns.length > 1) {
			Arrays.sort(soln.basisColumns);
		}
		for (int i = 1; i < soln.basisColumns.length; ++i) {
			if (soln.basisColumns[i] <= soln.basisColumns[i - 1]) {
				throw new LPException.LPErrorException(
						"duplicate column in basis");
			}
		}
		int nGood = 0;
		for (int i = 0; i < soln.basisColumns.length; ++i) {
			if (soln.basisColumns[i] < n) {
				++nGood;
			}
		}
		if (nGood<m) {
			// must adjust basis to be off slacks, should get here- but rounding could make this necessary
			final int[] sb = new int[nGood];
			nGood = 0;
			for (int i = 0; i < soln.basisColumns.length; ++i) {
				if (soln.basisColumns[i] < n) {
					sb[nGood] = soln.basisColumns[i];
					++nGood;
				}
			}
			final int[] rowset = new int[A.rows()];
			for (int i = 0; i < rowset.length; ++i) {
				rowset[i] = i;
			}
			final int[] eligableCols = new int[n];
			for(int i=0;i<n;++i) {
				eligableCols[i] = i;
			}
			// TODO: cut down the copies here!
			//final int[] nb = factory.matrixCopy(A.extractColumns(eligableCols)).colBasis(sb,minBasisEpsilon);
			final int[] nb = A.extractColumns(eligableCols).colBasis(sb,minBasisEpsilon);
			return nb;
		}
		return soln.basisColumns;
	}
	

	/**
	 * @param prob
	 *            well formed LPProb
	 * @param basis_in
	 *            (optional) valid initial basis
	 * @return x n-vector s.t. A x = b and x>=0 and c.x minimized allowed to
	 *         stop if A x = b, x>=0 c.x <=l, plus row and column basis for this solution
	 * @throws LPException
	 *             (if infeas or unbounded)
	 */
	@Override
	public <T extends Matrix<T>> LPSoln solve(final LPEQProb origProb, final int[] basis_in, final double tol,final int maxRounds, final LinalgFactory<T> factory)
			throws LPException {
		final long startTimeMS = System.currentTimeMillis();
		if (verbose > 0) {
			System.out.println("solve:");
			if (verbose > 1) {
				origProb.print();
			}
		}
		LPEQProb prob = origProb;
		final LPSoln soln;
		if((prob.A.rows()>0)&&(prob.A.cols()>0)) {
			// re-scale
			if(rescale) {
				final DenseVec newC = new DenseVec(origProb.c); // copy so we can re-scale without side effects
				prob = new LPEQProb(origProb.A, origProb.b.clone(),newC);
				final double scaleRange = 10.0;
				{
					final double[] rowTots = prob.A.sumAbsRowValues();
					final double[] scale = new double[rowTots.length];
					for(int i=0;i<prob.b.length;++i) {
						final double sumAbs = rowTots[i] + Math.abs(prob.b[i]);
						if((sumAbs>0)&&((sumAbs>=scaleRange*(prob.A.cols()+1.0))||(sumAbs<=(prob.A.cols()+1.0)/scaleRange))) {
							scale[i] = (prob.A.cols()+1.0)/sumAbs;
						} else {
							scale[i] = 1.0;
						}
						prob.b[i] *= scale[i];
					}
					prob = new LPEQProb(prob.A.rescaleRows(scale), prob.b, newC);
				}
				double sumAbs = 0.0;
				for(int j=0;j<prob.c.dim();++j) {
					sumAbs += Math.abs(prob.c.get(j));
				}
				if((sumAbs>0)&&((sumAbs>=scaleRange*prob.c.dim())||(sumAbs<=prob.c.dim()/scaleRange))) {
					final double scale = prob.c.dim()/sumAbs;
					for(int j=0;j<prob.c.dim();++j) {
						newC.set(j,newC.get(j)*scale);
					}
				}
			}
			final int[] basis0;
			if(null==basis_in) {
				basis0 = solvePhase1(prob.A, prob.b , prob.c, tol, maxRounds, factory);
			} else {
				basis0 = basis_in;
			}
			final int[] rb;
			if(basis0.length<=0) {
				rb = new int[0];
			} else if(basis0.length>=prob.A.rows()) { 
				rb = new int[prob.A.rows()];
				for(int i=0;i<rb.length;++i) {
					rb[i] = i;
				}
			} else {
				final PreMatrixI colSet = prob.A.extractColumns(basis0);
				rb = factory.matrixCopy(colSet).rowBasis(minBasisEpsilon); // TODO: get a better solution here, this is using nearly 1/2 of the time
			}
			if(rb.length>0) {
				if (rb.length != prob.A.rows()) {
					final PreMatrixI nA = prob.A.extractRows(rb);
					final double[] nb = Matrix.extract(prob.b,rb);
					prob = new LPEQProb(nA, nb, prob.c);
				}
				soln = rawSolve(prob, basis0, tol, maxRounds, factory, null);
				if ((soln == null) || (soln.primalSolution == null) || (soln.basisColumns == null)
						|| (soln.basisColumns.length != basis0.length)) {
					throw new LPException.LPErrorException(
							"bad basis back from phase1 raw solve");
				}
				soln.basisRows = rb;
			} else {
				soln = new LPSoln(new HVec(new int[0],new double[0]),new int[0],new int[0],0L);
			}
		} else {
			soln = new LPSoln(new HVec(new int[0],new double[0]),new int[0],new int[0],0L);
		}
		// check is needed as b-entries dropped out of the row set may be violated by solution, check causes throw
		LPEQProb.checkPrimFeas(origProb.A, origProb.b, soln.primalSolution, tol);
		// now check for zero columns (which can't enter a basis) for negative c (unbounded)
		{
			for(int j=0;j<origProb.c.dim();++j) {
				if(origProb.c.get(j)<0) {
					final SparseVec col = origProb.A.extractColumn(j);
					if(col.nzIndex()<0) {
						throw new LPException.LPUnboundedException("col " + j + " empty with c[j]=" + origProb.c.get(j));
					}
				}
			}
		}
		final long endTimeMS = System.currentTimeMillis();
		soln.reportedRunTimeMS = endTimeMS - startTimeMS;
		return soln;
	}
}