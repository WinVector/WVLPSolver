package com.winvector.lp;

import java.io.PrintStream;
import java.io.Serializable;

import com.winvector.linagl.LinalgFactory;
import com.winvector.linagl.Matrix;
import com.winvector.linagl.Vector;

/**
 * primal: min c.x: A x = b, x>=0 dual: max y.b: y A <= c y b = y A x <= c x (by
 * y A <=c, x>=0) , so y . b <= c . x at optimal y.b = c.x
 */
public final class LPEQProb<T extends Matrix<T>> implements Serializable {
	private static final long serialVersionUID = 1L;
	
	public final Matrix<T> A;
	public final Vector b;
	public final Vector c;
	

	/**
	 * @param A
	 *            matrix m-row by n-column matrix
	 * @param b
	 *            m-vector
	 * @param c
	 *            n-vector
	 * @throws LPException.LPMalformedException
	 *             if parameters don't match defs
	 */
	public LPEQProb(final Matrix<T> A_in, final Vector b_in, final Vector c_in)
			throws LPException.LPMalformedException {
		checkParams(A_in, b_in, c_in);
		A = A_in;
		b = b_in;
		c = c_in;
	}
	
	/**
	 * @param A_
	 *            matrix m-row by n-column matrix
	 * @param b
	 *            m-vector
	 * @param c_
	 *            n-vector
	 * @throws LPException.LPMalformedException
	 *             if parameters don't match defs
	 */
	public static <Z extends Matrix<Z>> void checkParams(Matrix<Z> A_, Vector b_, Vector c_)
			throws LPException.LPMalformedException {
		if ((A_ == null) || (b_ == null) || (c_ == null)
				|| (A_.rows() != b_.size()) || (A_.cols() != c_.size())) {
			String problem = "misformed problem";
			if (A_ == null) {
				problem = problem + " A_==null";
			}
			if (b_ == null) {
				problem = problem + " b==null";
			} else {
				if (A_ != null) {
					if (A_.rows() != b_.size()) {
						problem = problem + " A_.rows()(" + A_.rows()
								+ ")!=b.size()(" + b_.size() + ")";
					}
				}
			}
			if (c_ == null) {
				problem = problem + " c_==null";
			} else {
				if ((A_ != null) && (A_.rows() > 0)) {
					if (A_.cols() != c_.size()) {
						problem = problem + " A_.cols()(" + A_.cols()
								+ ")!=c.size()(" + c_.size() + ")";
					}
				}
			}
			throw new LPException.LPMalformedException(problem);
		}
	}


	public void print(PrintStream p) {
		p.println();
		p.println("x>=0");
		A.print(p);
		p.print(" * x = ");
		b.print(p);
		p.println();
		p.print("minimize x . ");
		c.print(p);
		p.println();
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
	public static <Z extends Matrix<Z>> Vector soln(final Matrix<Z> A, final Vector b, final int[] basis, final double tol)
			throws LPException {
		if ((A == null) || (b == null) || (basis == null) || (A.rows() <= 0)
				|| (A.rows() != b.size()) || (basis.length != A.rows())) {
			throw new LPException.LPErrorException("bad call to soln()");
		}
		if (A.rows() > A.cols()) {
			throw new LPException.LPErrorException("m>n in soln()");
		}
		final Matrix<Z> AP = A.extractColumns(basis);
		final Vector xp = AP.solve(b, false);
		final Vector x = b.newVector(A.cols());
		if (xp == null) {
			throw new LPException.LPErrorException("basis solution failed");
		}
		for (int i = 0; i < basis.length; ++i) {
			final double xpi = xp.get(i);
			if (xpi > 0) {
				x.set(basis[i], xpi);
			}
		}
		checkPrimFeas(A, b, x, tol);
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
	public static <Z extends Matrix<Z>> void checkPrimFeas(final Matrix<Z> A, final Vector b, final Vector x,
			double tol) throws LPException {
		if ((A == null) || (b == null) || (x == null)) {
			throw new LPException.LPInfeasibleException("null argument");
		}
		int m = A.rows();
		int n = A.cols();
		if ((b.size() != m) || (x.size() != n)) {
			throw new LPException.LPInfeasibleException(
					"wrong shaped vectors/matrix");
		}
		if ((tol <= 0.0)||Double.isNaN(tol)||Double.isInfinite(tol)) { 
			tol = 0.0;
		}
		for (int i = 0; i < n; ++i) {
			double xi = x.get(i);
			if (xi<-tol) {
				throw new LPException.LPInfeasibleException("negative entry");
			}
		}
		final Vector Ax = A.mult(x);
		for (int i = 0; i < m; ++i) {
			final double Axi = Ax.get(i);
			final double bi = b.get(i);
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
	public static <Z extends Matrix<Z>> void checkDualFeas(final Matrix<Z> A, final Vector c, final Vector y,
			double tol) throws LPException {
		if ((A == null) || (c == null) || (y == null)) {
			throw new LPException.LPInfeasibleException("null argument");
		}
		final int m = A.rows();
		final int n = A.cols();
		if ((c.size() != n) || (y.size() != m)) {
			throw new LPException.LPInfeasibleException(
					"wrong shaped vectors/matrix");
		}
		final Vector yA = A.multLeft(y);
		if ((tol <= 0.0)||Double.isNaN(tol)||Double.isInfinite(tol)) { 
			tol = 0.0;
		}
		for (int i = 0; i < n; ++i) {
			final double v = c.get(i) - yA.get(i);
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
	public static <Z extends Matrix<Z>> void checkPrimDualFeas(final Matrix<Z> A, final Vector b, final Vector c,
			final Vector x, final Vector y, double tol) throws LPException {
		if ((tol <= 0.0)||Double.isNaN(tol)||Double.isInfinite(tol)) { 
			tol = 0.0;
		}
		checkPrimFeas(A, b, x, tol);
		checkDualFeas(A, c, y, tol);
		final double yb = y.dot(b);
		final double cx = c.dot(x);
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
	public static <Z extends Matrix<Z>> void checkPrimDualOpt(final Matrix<Z> A, final Vector b, final Vector c, final Vector x,
			final Vector y, double tol) throws LPException {
		if ((tol <= 0.0)||Double.isNaN(tol)||Double.isInfinite(tol)) { 
			tol = 0.0;
		}
		checkPrimFeas(A, b, x, tol);
		checkDualFeas(A, c, y, tol);
		final double yb = y.dot(b);
		final double cx = c.dot(x);
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
	 * @return dual problem (converted to primal form)
	 * @throws LPException
	 *             (if infeas or unbounded)
	 * 
	 * dual: max y.b: y A <= c dual soln (useful for debugging): ( tran(A)
	 * -tran(A) I) ( y+ ) = c, ( y- ) ( s ) min (-b b 0).(y+ y- s), (y+ y- s) >=
	 * 0 sg(c) is a diagonal matrix with sg(c)_i,i = 1 if c_i>=0, -1 otherwise
	 */
	LPEQProb<T> dual() throws LPException {
		final int m = A.rows();
		final int n = A.cols();
		final Vector cp = b.newVector(2 * m + n);
		final Matrix<T> AP = A.newMatrix(n, cp.size(),A.sparseRep());
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				final double aji = A.get(j, i);
				if(0.0!=aji) {
					AP.set(i, j, A.get(j, i));
					AP.set(i, j + m, -A.get(j, i));
				}
			}
			AP.set(i, i + 2 * m, 1.0);
		}
		for (int i = 0; i < m; ++i) {
			cp.set(i, -b.get(i));
			cp.set(i + m, b.get(i));
		}
		return new LPEQProb<T>(AP, c, cp);
	}

	/**
	 * @param p
	 *            primal optimal solution
	 * @param tol
	 *            tolerance for comparisions
	 * @param factory 
	 * @return dual-optimal solution
	 */
	public Vector inspectForDual(final LPSoln<T> p, final double tol, final LinalgFactory<T> factory) throws LPException {
		checkPrimFeas(A, b, p.x, tol);
		// we now have a list of equality constraints to work with
		try {
			// least squares solve sub-system
			final int[] rb = A.rowBasis(null,1.0e-5);
			if ((rb == null) || (rb.length <= 0)) {
				final Vector y = b.newVector(b.size());
				checkPrimDualFeas(A, b, c, p.x, y, tol);
				checkPrimDualOpt(A, b, c, p.x, y, tol);
				return y;
			}
			final Matrix<T> eqmat = factory.newMatrix(p.basis.length+1, p.basis.length,A.sparseRep());
			final Vector eqvec = factory.newVector(p.basis.length+1);
			// put in obj-relation
			eqmat.setRow(0,b.extract(rb));
			eqvec.set(0,c.dot(p.x));
			// put in all complementary slackness relns
			// Schrijver p. 95
			for (int bi = 0; bi < p.basis.length; ++bi) {
				int i = p.basis[bi];
				eqmat.setRow(bi+1,A.extractColumn(i).extract(rb));
				eqvec.set(bi+1,c.get(i));
			}
			final Vector yr = eqmat.solve(eqvec, true);
			final Vector y = b.newVector(b.size());
			if (yr != null) {
				int j = -1;
				while ((j = yr.nextIndex(j)) >= 0) {
					y.set(rb[j], yr.get(j));
				}
			}
			checkPrimDualFeas(A, b, c, p.x, y, tol);
			checkPrimDualOpt(A, b, c, p.x, y, tol);
			return y;
		} catch (Exception e) {
			System.out.println("caughtZ: " + e);
			return null;
		}
	}

	/**
	 * @param solver
	 *            solver
	 * @param basis_in
	 *            (optional) valid initial basis
	 * @return y m-vector s.t. y A <= c and y.b maximized
	 * @throws LPException
	 *             (if infeas or unbounded)
	 */
	public LPSoln<T> solveDual(final LPSolver<T> solver, final int[] basis_in, final double tol, final int maxRounds)
			throws LPException {
		int m = A.rows();
		final LPEQProb<T> dual = dual();
		final LPSoln<T> s = solver.solve(dual, basis_in, tol,maxRounds);
		final Vector y = b.newVector(m);
		for (int i = 0; i < m; ++i) {
			y.set(i, s.x.get(i) - s.x.get(i + m));
		}
		checkDualFeas(A, c, y, tol);
		return new LPSoln<T>(y, s.basis);
	}

	/**
	 * solves primal and dual to prove solution
	 * 
	 * @param solver
	 *            solver
	 * @throws LPException
	 *             (if infeas or unbounded)
	 */
	public LPSoln<T> solveDebugByInspect(final LPSolver<T> solver, final double tol, final LinalgFactory<T> factory, final int maxRounds)
			throws LPException {
		final LPSoln<T> primSoln = solver.solve(this, null, 1.0e-6,maxRounds);
		final Vector y2 = inspectForDual(primSoln, tol, factory);
		checkPrimDualOpt(A, b, c, primSoln.x, y2, tol);
		//LPSoln dualSoln = solveDual(solver,null);
		//checkPrimDualOpt(primSoln.x,dualSoln.x);
		return primSoln;
	}


	public int nvars() {
		if (A != null) {
			return A.cols();
		}
		if (c != null) {
			return c.size();
		}
		return 0;
	}


	public void print() {
		print(System.out);
	}


	public double primalValue(final Vector x) {
		return c.dot(x);
	}


	public double dualValue(final Vector y) {
		return b.dot(y);
	}


	/**
	 * solves primal and dual to prove soluton
	 * 
	 * @param solver
	 *            solver
	 * @throws LPException
	 *             (if infeas or unbounded)
	 */
	public LPSoln<T> solveDebug(final LPSolver<T> solver, final double tol, final int maxRounds) throws LPException {
		final LPSoln<T> primSoln = solver.solve(this, null, tol,maxRounds);
		final LPSoln<T> dualSoln = solveDual(solver, null, tol,maxRounds);
		checkPrimDualOpt(A, b, c, primSoln.x, dualSoln.x, tol);
		return primSoln;
	}
}