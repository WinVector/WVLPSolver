package com.winvector.lp;

import java.util.Random;

import com.winvector.linagl.HVec;
import com.winvector.linagl.LinalgFactory;
import com.winvector.linagl.Matrix;
import com.winvector.linagl.PreMatrixI;
import com.winvector.linagl.PreVecI;
import com.winvector.linagl.SparseVec;
import com.winvector.lp.LPException.LPMalformedException;
import com.winvector.lp.impl.RandomOrder;


/**
 * primal: min c.x: A x = b, x>=0 
 * dual: max y.b: y A <= c 
 * y b = y A x <= c x (by A <=c, x>=0) , so y . b <= c . x at optimal y.b = c.x
 */
public final class LPEQProb extends LPProbBase implements LPEQProbI {
	private static final long serialVersionUID = 1L;
	

	public LPEQProb(final PreMatrixI A_in, final double[] b_in, final PreVecI c_in)
			throws LPException.LPMalformedException {
		super(A_in,b_in,c_in,"=");
	}

	/**
	 * @param A
	 *            matrix m-row by n-column matrix m <=n
	 * @param b
	 *            m-vector
	 * @param basis
	 *            m-vector that is a valid starting basis
	 * @param tol
	 *            tolerance for comparisons
	 * @return x s.t. x(basis) = A(basis)^-1 b, x zero in other positions, x>=0 (
	 *         A(basis) = square matrix of basis columns x(basis) = vector with
	 *         entries selected by basis)
	 * @throws LPException
	 *             on bad data
	 */
	public static <T extends Matrix<T>> HVec primalSoln(final PreMatrixI A, final double[] b, final int[] basis, final double tol, final LinalgFactory<T> factory)
			throws LPException {
		if ((A == null) || (b == null) || (basis == null) || (A.rows() <= 0)
				|| (A.rows() != b.length) || (basis.length != A.rows())) {
			throw new LPException.LPErrorException("bad call to soln()");
		}
		if (A.rows() > A.cols()) {
			throw new LPException.LPErrorException("m>n in soln()");
		}
		final Matrix<T> AP = A.extractColumns(basis,factory);
		final double[] xp = AP.solve(b);
		if (xp == null) {
			throw new LPException.LPErrorException("basis solution failed");
		}
		final HVec x = new HVec(basis,xp);
		checkPrimFeas(A, b, x, tol);
		return x;
	}
	
	public static <T extends Matrix<T>> HVec primalSoln(final LPEQProbI prob, final int[] basis, final LinalgFactory<T> factory)
			throws LPException {
		final Matrix<T> AP = prob.extractColumns(basis,factory);
		final double[] xp = AP.solve(prob.b());
		if (xp == null) {
			throw new LPException.LPErrorException("basis solution failed");
		}
		final HVec x = new HVec(basis,xp); 
		return x;
	}

	/**
	 * @param A
	 *            matrix m-row by n-column matrix
	 * @param b
	 *            m-vector
	 * @param x
	 *            n-vector s.t. A x = b and x>=0
	 * @param tol
	 *            tollerance for comparisons
	 * @throws LPException
	 *             (if infeas ill-formed)
	 */
	public static void checkPrimFeas(final PreMatrixI A, final double[] b, final HVec x,
			double tol) throws LPException {
		if ((A == null) || (b == null) || (x == null)) {
			throw new LPException.LPInfeasibleException("null argument");
		}
		int m = A.rows();
		if ((b.length != m) ) {
			throw new LPException.LPInfeasibleException(
					"wrong shaped vectors/matrix");
		}
		if ((tol <= 0.0)||Double.isNaN(tol)||Double.isInfinite(tol)) { 
			tol = 0.0;
		}
		final int nindices = x.nIndices();
		for (int ii = 0; ii < nindices; ++ii) {
			final double xi = x.value(ii);
			if (xi<-tol) {
				throw new LPException.LPInfeasibleException("negative entry");
			}
		}
		final double[] Ax = A.mult(x);
		for (int i = 0; i < m; ++i) {
			final double Axi = Ax[i];
			final double bi = b[i];
			final double diffi = Math.abs(Axi-bi);
			if (diffi>tol) {
				throw new LPException.LPInfeasibleException(
						"equality violated " + diffi);
			}
		}
		// all is well, do nothing
	}

	/**
	 * @param A
	 *            matrix m-row by n-column matrix
	 * @param c
	 *            n-vector
	 * @param y
	 *            m-vector s.t. y A <= c
	 * @param tol
	 *            tolerance for comparisions
	 * @throws LPException
	 *             (if infeas ill-formed)
	 */
	public static void checkDualFeas(final PreMatrixI A, final PreVecI c, final double[] y,
			double tol) throws LPException {
		if ((A == null) || (c == null) || (y == null)) {
			throw new LPException.LPInfeasibleException("null argument");
		}
		final int m = A.rows();
		final int n = A.cols();
		if ((c.dim() != n) || (y.length != m)) {
			throw new LPException.LPInfeasibleException(
					"wrong shaped vectors/matrix");
		}
		final double[] yA = A.multLeft(y);
		if ((tol <= 0.0)||Double.isNaN(tol)||Double.isInfinite(tol)) { 
			tol = 0.0;
		}
		for (int i = 0; i < n; ++i) {
			final double v = c.get(i) - yA[i];
			if (v<-tol) {
				throw new LPException.LPInfeasibleException(
						"inequality violated");
			}
		}
		// all is well, do nothing
	}

	/**
	 * @param A
	 *            matrix m-row by n-column matrix
	 * @param b
	 *            m-vector
	 * @param c
	 *            n-vector
	 * @param x
	 *            n-vector s.t. A x = b and x>=0
	 * @param y
	 *            m-vector s.t. y A <= c
	 * @param tol
	 *            tolerance for comparisions
	 * @throws LPException
	 *             (if infeas ill-formed, or y.b > c.x)
	 */
	public static void checkPrimDualFeas(final PreMatrixI A, final double[] b, final PreVecI c,
			final HVec x, final double[] y, double tol) throws LPException {
		if ((tol <= 0.0)||Double.isNaN(tol)||Double.isInfinite(tol)) { 
			tol = 0.0;
		}
		checkPrimFeas(A, b, x, tol);
		checkDualFeas(A, c, y, tol);
		final double yb = Matrix.dot(y,b);
		final double cx = x.dot(c);
		final double d = cx - yb;
		if (d<-tol) {
			// not possible by above checks, but helps us debug above checks
			throw new LPException.LPInfeasibleException("yb>cx");
		}
	}

	/**
	 * @param A
	 *            matrix m-row by n-column matrix
	 * @param b
	 *            m-vector
	 * @param c
	 *            n-vector
	 * @param x
	 *            n-vector s.t. A x = b and x>=0
	 * @param y
	 *            m-vector s.t. y A <= c
	 * @param tol
	 *            tolerance for comparisions
	 * @throws LPException
	 *             (if infeas ill-formed, or y.b != c.x)
	 */
	public static void checkPrimDualOpt(final PreMatrixI A, final double[] b, final PreVecI c, 
			final HVec x,
			final double[] y, double tol) throws LPException {
		if ((tol <= 0.0)||Double.isNaN(tol)||Double.isInfinite(tol)) { 
			tol = 0.0;
		}
		checkPrimFeas(A, b, x, tol);
		checkDualFeas(A, c, y, tol);
		final double yb = Matrix.dot(y,b);
		final double cx = x.dot(c);
		final double d = cx - yb;
		if (d < -tol) {
			// not possible by above checks, but helps us debug above checks
			throw new LPException.LPInfeasibleException("yb>cx");
		}
		final double gap = yb - cx;
		if (Math.abs(gap)>tol) {
			//System.out.println("yb: " + yb + " cx: " + cx);
			throw new LPException.LPInfeasibleException("not optimal");
		}
	}

	

	/**
	 * @param p
	 *            primal optimal solution
	 * @param tol
	 *            tolerance for comparisions
	 * @param factory 
	 * @return dual-optimal solution
	 */
	 public <T extends Matrix<T>> double[] dualSolution(final LPSoln p, final double tol, final LinalgFactory<T> factory) throws LPException {
		 checkPrimFeas(A, b, p.primalSolution, tol);
		 // we now have a list of equality constraints to work with
		 final int[] rb;
		 if(p.basisRows!=null) {
			 rb = p.basisRows;
		 } else {
			 rb = A.matrixCopy(factory).rowBasis(1.0e-5);
		 }
		 if ((rb == null) || (rb.length <= 0)) {
			 final double[] y = new double[b.length];
			 checkPrimDualFeas(A, b, c, p.primalSolution, y, tol);
			 checkPrimDualOpt(A, b, c, p.primalSolution, y, tol);
			 return y;
		 }
		 final T eqmat = factory.newMatrix(p.basisColumns.length, p.basisColumns.length,true);
		 final double[] eqvec = new double[p.basisColumns.length];
		 // put in all complementary slackness relns
		 // Schrijver p. 95
		 final Object extractTemps = A.buildExtractTemps();
		 for (int bi = 0; bi < p.basisColumns.length; ++bi) {
			 int i = p.basisColumns[bi];
			 eqmat.setRow(bi,Matrix.extract(A.extractColumn(i,extractTemps).toDense(),rb));
			 eqvec[bi] = c.get(i);
		 }
		 final double[] yr = eqmat.solve(eqvec);
		 final double[] y = new double[b.length];
		 if (yr != null) {
			 for(int j=0;j<yr.length;++j) {
				 y[rb[j]] = yr[j];
			 }
		 }
		 checkPrimDualFeas(A, b, c, p.primalSolution, y, tol);
		 checkPrimDualOpt(A, b, c, p.primalSolution, y, tol);
		 return y;
	 }

	public int nvars() {
		if (A != null) {
			return A.cols();
		}
		if (c != null) {
			return c.dim();
		}
		return 0;
	}
	
	@Override
	public LPEQProb eqForm() throws LPMalformedException {
		return this;
	}


	/**
	 * solves primal and dual to prove soluton
	 * 
	 * @param solver
	 *            solver
	 * @throws LPException
	 *             (if infeas or unbounded)
	 */
	public <T extends Matrix<T>> LPSoln solveDebug(final LPSolver solver, final double tol, final int maxRounds, final LinalgFactory<T> factory) throws LPException {
		final LPSoln primSoln = solver.solve(this, null, tol, maxRounds, factory);
		final double[] dualSoln = dualSolution(primSoln, tol,factory);
		checkPrimDualOpt(A, b, c, primSoln.primalSolution, dualSoln, tol);
		return primSoln;
	}

	// Abstract LPEQProb methods
	@Override
	public int rows() {
		return A.rows();
	}
	
	@Override
	public Object buildExtractTemps() {
		return A.buildExtractTemps();
	}

	@Override
	public SparseVec extractColumn(final int j, final Object extractTemps) {
		return A.extractColumn(j,extractTemps);
	}

	@Override
	public <T extends Matrix<T>> T extractColumns(final int[] basis,
			final LinalgFactory<T> factory) {
		return A.extractColumns(basis, factory);
	}

	@Override
	public double c(final int i) {
		return c.get(i);
	}

	@Override
	public double[] b() {
		return b;
	}

	@Override
	public InspectionOrder buildOrderTracker(final Random rand) {
		return new RandomOrder(A.cols(),rand);
	}
}