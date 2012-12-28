package com.winvector.lp;

import java.io.PrintStream;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.text.NumberFormat;

import com.winvector.linagl.LinalgFactory;
import com.winvector.linagl.Matrix;
import com.winvector.linagl.PreMatrix;
import com.winvector.sparse.ColumnMatrix;


/**
 * primal: min c.x: A x = b, x>=0 dual: max y.b: y A <= c y b = y A x <= c x (by
 * y A <=c, x>=0) , so y . b <= c . x at optimal y.b = c.x
 */
public final class LPEQProb implements Serializable {
	private static final long serialVersionUID = 1L;
	
	public final ColumnMatrix A;
	public final double[] b;
	public final double[] c;
	

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
	public LPEQProb(final ColumnMatrix A_in, final double[] b_in, final double[] c_in)
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
	public static void checkParams(final PreMatrix A_, final double[] b_, final double[] c_)
			throws LPException.LPMalformedException {
		if ((A_ == null) || (b_ == null) || (c_ == null)
				|| (A_.rows() != b_.length) || (A_.cols() != c_.length)) {
			String problem = "misformed problem";
			if (A_ == null) {
				problem = problem + " A_==null";
			}
			if (b_ == null) {
				problem = problem + " b==null";
			} else {
				if (A_ != null) {
					if (A_.rows() != b_.length) {
						problem = problem + " A_.rows()(" + A_.rows()
								+ ")!=b.length(" + b_.length + ")";
					}
				}
			}
			if (c_ == null) {
				problem = problem + " c_==null";
			} else {
				if ((A_ != null) && (A_.rows() > 0)) {
					if (A_.cols() != c_.length) {
						problem = problem + " A_.cols()(" + A_.cols()
								+ ")!=c.length(" + c_.length + ")";
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
		p.println(Matrix.toString(b));
		p.print("minimize x . ");
		p.println(Matrix.toString(c));
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
	public static <T extends Matrix<T>> double[] primalSoln(final PreMatrix A, final double[] b, final int[] basis, final double tol, final LinalgFactory<T> factory)
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
		final double[] x = new double[A.cols()];
		if (xp == null) {
			throw new LPException.LPErrorException("basis solution failed");
		}
		for (int i = 0; i < basis.length; ++i) {
			final double xpi = xp[i];
			x[basis[i]] = xpi;
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
	public static void checkPrimFeas(final PreMatrix A, final double[] b, final double[] x,
			double tol) throws LPException {
		if ((A == null) || (b == null) || (x == null)) {
			throw new LPException.LPInfeasibleException("null argument");
		}
		int m = A.rows();
		int n = A.cols();
		if ((b.length != m) || (x.length != n)) {
			throw new LPException.LPInfeasibleException(
					"wrong shaped vectors/matrix");
		}
		if ((tol <= 0.0)||Double.isNaN(tol)||Double.isInfinite(tol)) { 
			tol = 0.0;
		}
		for (int i = 0; i < n; ++i) {
			final double xi = x[i];
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
	public static void checkDualFeas(final PreMatrix A, final double[] c, final double[] y,
			double tol) throws LPException {
		if ((A == null) || (c == null) || (y == null)) {
			throw new LPException.LPInfeasibleException("null argument");
		}
		final int m = A.rows();
		final int n = A.cols();
		if ((c.length != n) || (y.length != m)) {
			throw new LPException.LPInfeasibleException(
					"wrong shaped vectors/matrix");
		}
		final double[] yA = A.multLeft(y);
		if ((tol <= 0.0)||Double.isNaN(tol)||Double.isInfinite(tol)) { 
			tol = 0.0;
		}
		for (int i = 0; i < n; ++i) {
			final double v = c[i] - yA[i];
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
	public static void checkPrimDualFeas(final PreMatrix A, final double[] b, final double[] c,
			final double[] x, final double[] y, double tol) throws LPException {
		if ((tol <= 0.0)||Double.isNaN(tol)||Double.isInfinite(tol)) { 
			tol = 0.0;
		}
		checkPrimFeas(A, b, x, tol);
		checkDualFeas(A, c, y, tol);
		final double yb = Matrix.dot(y,b);
		final double cx = Matrix.dot(c,x);
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
	public static void checkPrimDualOpt(final PreMatrix A, final double[] b, final double[] c, final double[] x,
			final double[] y, double tol) throws LPException {
		if ((tol <= 0.0)||Double.isNaN(tol)||Double.isInfinite(tol)) { 
			tol = 0.0;
		}
		checkPrimFeas(A, b, x, tol);
		checkDualFeas(A, c, y, tol);
		final double yb = Matrix.dot(y,b);
		final double cx = Matrix.dot(c,x);
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
		 final T fullA = A.matrixCopy(factory);
		 final int[] rb = fullA.rowBasis(null,1.0e-5);
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
		 for (int bi = 0; bi < p.basisColumns.length; ++bi) {
			 int i = p.basisColumns[bi];
			 eqmat.setRow(bi,Matrix.extract(fullA.extractColumn(i),rb));
			 eqvec[bi] = c[i];
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
			return A.cols;
		}
		if (c != null) {
			return c.length;
		}
		return 0;
	}


	public void print() {
		print(System.out);
	}


	public double primalValue(final double[] x) {
		return Matrix.dot(c,x);
	}


	public double dualValue(final double[] y) {
		return Matrix.dot(b,y);
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

	/**
	 * print out in CPLEX problem format
	 * @param p
	 */
	public void printCPLEX(final PrintStream p) {
		final NumberFormat vnf = new DecimalFormat("00000");
		final NumberFormat vvf = new DecimalFormat("#.######E0");
		p.println("\\* WVLPSovler com.winvector.lp.LPEQProb see: http://www.win-vector.com/blog/2012/11/yet-another-java-linear-programming-library/ *\\");
		p.println();
		p.println("Minimize");
		p.print("\tvalue: ");
		{
			boolean first = true;
			for(int j=0;j<c.length;++j) {
				final double cj = c[j];
				if(Math.abs(cj)!=0) {
					final String valCStr = vvf.format(cj);
					if(!first) {
						if(valCStr.charAt(0)!='-') {
							p.print(" +");
						} else {
							p.print(" ");
						}
					} else {
						first = false;
					}
					p.print(valCStr + " " + "x" + vnf.format(j));
				}
			}
		}
		p.println();
		p.println();
		p.println("Subject To");
		for(int i=0;i<b.length;++i) {
			int nnz = 0;
			for(int j=0;j<c.length;++j) {
				final double aij = A.get(i, j);
				if(Math.abs(aij)!=0) {
					++nnz;
				}
			}
			if(nnz>0) {
				p.print("\teq" + vnf.format(i) + ":\t");
				boolean first = true;
				for(int j=0;j<c.length;++j) {
					final double aij = A.get(i, j);
					if(Math.abs(aij)!=0) {
						final String valStr = vvf.format(aij);
						if(!first) {
							if(valStr.charAt(0)!='-') {
								p.print(" +");
							} else {
								p.print(" ");
							}
						} else {
							first = false;
						}
						p.print(valStr + " " + "x" + vnf.format(j));
					}
				}
				p.println(" = " + vvf.format(b[i]));
			}
		}
		p.println();
		p.println("Bounds");
		for(int j=0;j<c.length;++j) {
			p.println("\t0 <= x" + vnf.format(j));
		}
		p.println();
		p.println("End");
		p.println();
		p.println("\\* eof *\\");
	}
}