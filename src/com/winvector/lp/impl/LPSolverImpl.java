package com.winvector.lp.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import com.winvector.linagl.LinalgFactory;
import com.winvector.linagl.Matrix;
import com.winvector.linagl.PreMatrix;
import com.winvector.lp.EarlyExitCondition;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPException;
import com.winvector.lp.LPSoln;
import com.winvector.lp.LPSolver;
import com.winvector.sparse.ColumnMatrix;
import com.winvector.sparse.SparseVec;

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
	public static void checkParams(final PreMatrix A, final double[] b, final double[] c, final int[] basis)
			throws LPException.LPMalformedException {
		if ((A == null) || (b == null) || (c == null) || (basis == null)
				|| (A.rows() <= 0) || (A.rows() != b.length)
				|| (A.rows() != basis.length) || (A.cols() != c.length)) {
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
					if (A.cols() != c.length) {
						problem = problem + " A.cols()(" + A.cols()
								+ ")!=c.length(" + c.length + ")";
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

	/**
	 * @param A
	 *            full row-rank m by n matrix
	 * @param basis
	 *            column basis
	 * @param hint
	 *            (optional) n-vector
	 * @return basic solution to A x = b x>=0 or null (from basis)
	 */
	private static <T extends Matrix<T>> LPSoln tryBasis(final PreMatrix A, final int[] basis, final double[] b, final LinalgFactory<T> factory) {
		if ((basis == null) || (basis.length != A.rows())) {
			return null;
		}
		double[] x = null;
		try {
			final Matrix<T> AP = A.extractColumns(basis,factory);
			x = AP.solve(b);
		} catch (Exception e) {
		}
		if (x == null) {
			return null;
		}
		for(int j=0;j<x.length;++j) {
			if (x[j] < 0) {
				return null;
			}
		}
		return new LPSoln(x, basis);
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
	protected abstract <T extends Matrix<T>> LPSoln rawSolve(LPEQProb prob, int[] basis0,
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
	private <T extends Matrix<T>> int[] solvePhase1(final ColumnMatrix A, final double[] b, final double[] cin, final double tol, 
			final int maxRounds, final LinalgFactory<T> factory) 
			throws LPException {
		final int m = A.rows;
		final int n = A.cols;
		final double[] c = new double[n + m];
		final ColumnMatrix AP;
		{
			final ArrayList<SparseVec> slackCols = new ArrayList<SparseVec>(m);
			for(int i=0;i<m;++i) {
				slackCols.add(new SparseVec(m,i,b[i]>=0?1.0:-1.0));
			}
			AP = A.addColumns(slackCols);
		}
		for(int i=n;i<m+n;++i) {
			c[i] = 1.0;
		}
		if(null!=cin) {
			// hint at actual objective fn
			double sumAbscin = 0.0;
			for(int i=0;i<n;++i) {
				sumAbscin += Math.abs(cin[i]);
			}
			final double scale = 1.0e-7/(1.0+sumAbscin);
			for(int i=0;i<n;++i) {
				c[i] = scale*cin[i];
			}
		}
		final LPEQProb p1prob = new LPEQProb(AP, b, c);
		final int[] basis0 = new int[m];
		for(int i=0;i<m;++i) {
			basis0[i] = n + i;
		}
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
				|| (soln.basisColumns.length != basis0.length) || (soln.primalSolution == null)
				|| (soln.primalSolution.length != c.length)) {
			throw new LPException.LPErrorException(
					"bad basis back from phase1 raw solve");
		}
		// check objective value is zero
		final double v = Matrix.dot(c,soln.primalSolution);
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
			final int[] rowset = new int[n];
			for (int i = 0; i < rowset.length; ++i) {
				rowset[i] = i;
			}
			// TODO: cut down the copies here!
			final int[] nb = A.matrixCopy(factory).extractRows(rowset,factory).colBasis(sb,minBasisEpsilon);
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
		// get rid of degenerate cases
		//System.out.println("start rb1");
		final int[] rb = origProb.A.matrixCopy(factory).rowBasis(null,minBasisEpsilon);
		if ((rb == null) || (rb.length <= 0)) {
			//solving 0 x = b
			if (!Matrix.isZero(origProb.b)) {
				throw new LPException.LPInfeasibleException(
						"linear relaxation incosistent");
			}
			for (int i = 0; i < origProb.c.length; ++i) {
				if (origProb.c[i] < 0) {
					throw new LPException.LPUnboundedException(
							"unbounded minimum solving 0 x = 0");
				}
			}
			final double[] x = new double[origProb.A.cols];
			int[] b = new int[x.length];
			for (int i = 0; i < b.length; ++i) {
				b[i] = i;
			}
			return new LPSoln(x, b, rb);
		}
		LPEQProb prob = null;
		// select out irredundant rows
		if (rb.length != origProb.A.rows) {
			final ColumnMatrix nA = origProb.A.extractRows(rb);
			final double[] nb = Matrix.extract(origProb.b,rb);
			prob = new LPEQProb(nA, nb, origProb.c.clone());
		} else {
			prob = new LPEQProb(origProb.A, origProb.b.clone(), origProb.c.clone());
		}
		// deal with square system
		if (prob.A.rows >= prob.A.cols) {
			final double[] x = prob.A.matrixCopy(factory).solve(prob.b);
			if (x == null) {
				throw new LPException.LPInfeasibleException(
						"linear problem infeasible");
			}
			LPEQProb.checkPrimFeas(prob.A, prob.b, x, tol);
			if (prob != origProb) {
				LPEQProb.checkPrimFeas(origProb.A, origProb.b, x, tol);
			}
			int[] b = new int[x.length];
			for (int i = 0; i < b.length; ++i) {
				b[i] = i;
			}
			return new LPSoln(x, b, rb);
		}
		// re-scale
		final double scaleRange = 10.0;
		{
			final double[] rowTots = prob.A.sumAbsRowValues();
			final double[] scale = new double[rowTots.length];
			for(int i=0;i<prob.b.length;++i) {
				final double sumAbs = rowTots[i] + Math.abs(prob.b[i]);
				if((sumAbs>0)&&((sumAbs>=scaleRange*(prob.A.cols+1.0))||(sumAbs<=(prob.A.cols+1.0)/scaleRange))) {
					scale[i] = (prob.A.cols+1.0)/sumAbs;
				} else {
					scale[i] = 1.0;
				}
				prob.b[i] *= scale[i];
			}
			prob = new LPEQProb(prob.A.rescaleRows(scale), prob.b, prob.c);
		}
		{
			double sumAbs = 0.0;
			for(int j=0;j<prob.c.length;++j) {
				sumAbs += Math.abs(prob.c[j]);
			}
			if((sumAbs>0)&&((sumAbs>=scaleRange*prob.c.length)||(sumAbs<=prob.c.length/scaleRange))) {
				final double scale = prob.c.length/sumAbs;
				for(int j=0;j<prob.c.length;++j) {
					prob.c[j] *= scale;
				}
			}
		}
		// work on basis
		int[] basis0 = null;
		if ((basis_in != null) && (basis_in.length == prob.A.rows)) {
			try {
				if (verbose > 0) {
					System.out.println("import basis");
				}
				basis0 = new int[basis_in.length];
				for (int i = 0; i < basis0.length; ++i) {
					basis0[i] = basis_in[i];
				}
				final double[] x0 = LPEQProb.primalSoln(prob.A, prob.b, basis0, tol, factory);
				LPEQProb.checkPrimFeas(prob.A, prob.b, x0, tol);
			} catch (Exception e) {
				basis0 = null;
				System.out.println("caught: " + e);
			}
		}
		if (basis0 == null) {
			if (verbose > 0) {
				System.out.println("phase1");
			}
			basis0 = solvePhase1(prob.A, prob.b , prob.c, tol, maxRounds, factory);
		}
		if (verbose > 0) {
			System.out.println("phase2");
		}
		final LPSoln soln;
		if (Matrix.isZero(prob.c)) {
			// no objective function, any basis will do
			soln = tryBasis(prob.A, basis0, prob.b, factory);
		} else {
			soln = rawSolve(prob, basis0, tol, maxRounds, factory, null);
			if ((soln == null) || (soln.primalSolution == null) || (soln.basisColumns == null)
				|| (soln.basisColumns.length != basis0.length)) {
				throw new LPException.LPErrorException(
						"bad basis back from phase1 raw solve");
			}
		}
		//System.out.println("phase1steps " + phase1StepsTaken + ", phase2 steps " + soln.stepsTaken);
		LPEQProb.checkPrimFeas(prob.A, prob.b, soln.primalSolution, tol);
		if (prob != origProb) {
			LPEQProb.checkPrimFeas(origProb.A, origProb.b, soln.primalSolution, tol);
		}
		soln.basisRows = rb;
		return soln;
	}
}