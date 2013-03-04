package com.winvector.lp.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import com.winvector.linalg.ColumnMatrix;
import com.winvector.linalg.DenseVec;
import com.winvector.linalg.LinalgFactory;
import com.winvector.linalg.Matrix;
import com.winvector.linalg.PreMatrixI;
import com.winvector.linalg.PreVecI;
import com.winvector.linalg.SparseVec;
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
		final Set<Integer> seen = new HashSet<Integer>();
		for (int i = 0; i < basis.length; ++i) {
			if ((basis[i] < 0) || (basis[i] >= n)) {
				throw new LPException.LPMalformedException(
						"out of range column in basis");
			}
			Integer key = new Integer(basis[i]);
			if (seen.contains(key)) {
				throw new LPException.LPMalformedException(
						"duplicate column in basis");
			}
		}
	}

	/**
	 * @param ncols
	 *            the number of columns we are dealing with
	 * @param basis
	 *            a list of columns
	 * @return [0..ncols-1] set-minus basis
	 */
	static int[] complementaryColumns(int ncols, int[] basis) {
		final Set<Integer> seen = new HashSet<Integer>();
		if (basis != null) {
			for (int i = 0; i < basis.length; ++i) {
				if ((basis[i] >= 0) && (basis[i] < ncols)) {
					seen.add(new Integer(basis[i]));
				}
			}
		}
		int[] r = new int[ncols - seen.size()];
		int j = 0;
		for (int i = 0; (i < ncols) && (j < r.length); ++i) {
			Integer key = new Integer(i);
			if (!seen.contains(key)) {
				r[j] = i;
				++j;
			}
		}
		return r;
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
		{
			Arrays.fill(basis0,-1);
			final Object extractTemps = A.buildExtractTemps();
			for(int j=0;j<n;++j) {
				final SparseVec col = A.extractColumn(j,extractTemps);
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
		final ColumnMatrix AP = new ColumnMatrix(A).addColumns(artificialSlackCols);
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
			// TODO: cut down the copies here!
			final int[] eligableCols = new int[n];
			for(int i=0;i<n;++i) {
				eligableCols[i] = i;
			}
			final int[] nb = A.matrixCopy(factory).extractColumns(eligableCols, factory).colBasis(sb,minBasisEpsilon);
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
		if (verbose > 0) {
			System.out.println("solve:");
			if (verbose > 1) {
				origProb.print();
			}
		}
		LPEQProb prob = origProb;
		// re-scale
		if(rescale) {
			final DenseVec newC = new DenseVec(origProb.c); // copy so we can re-scale without side effects
			prob = new LPEQProb(origProb.A, origProb.b.clone(),newC);
			final double scaleRange = 10.0;
			{
				final ColumnMatrix probA = new ColumnMatrix(prob.A);
				final double[] rowTots = probA.sumAbsRowValues();
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
				prob = new LPEQProb(probA.rescaleRows(scale), prob.b, newC);
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
		{
			if(basis0.length>=prob.A.rows()) {
				rb = new int[prob.A.rows()];
				for(int i=0;i<rb.length;++i) {
					rb[i] = i;
				}
			} else {
				final T colSet = prob.A.extractColumns(basis0, factory);
				rb = colSet.rowBasis(minBasisEpsilon); // TODO: get a better solution here, this is using nearly 1/2 of the time
			}
		}
		if (rb.length != prob.A.rows()) {
			 final ColumnMatrix nA = new ColumnMatrix(prob.A).extractRows(rb);
			 final double[] nb = Matrix.extract(prob.b,rb);
			 prob = new LPEQProb(nA, nb, prob.c);
		}
		final LPSoln soln = rawSolve(prob, basis0, tol, maxRounds, factory, null);
		if ((soln == null) || (soln.primalSolution == null) || (soln.basisColumns == null)
				|| (soln.basisColumns.length != basis0.length)) {
			throw new LPException.LPErrorException(
					"bad basis back from phase1 raw solve");
		}
		//System.out.println("phase1steps " + phase1StepsTaken + ", phase2 steps " + soln.stepsTaken);
		LPEQProb.checkPrimFeas(origProb.A, origProb.b, soln.primalSolution, tol);
		soln.basisRows = rb;
		for(int i=0;i<rb.length;++i) {
			rb[i] = i;
		}
		return soln;
	}
}