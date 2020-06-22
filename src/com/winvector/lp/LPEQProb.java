package com.winvector.lp;

import java.io.PrintStream;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Random;

import com.winvector.linalg.LinalgFactory;
import com.winvector.linalg.Matrix;
import com.winvector.linalg.PreMatrixI;
import com.winvector.linalg.PreVecI;
import com.winvector.linalg.sparse.HVec;
import com.winvector.linalg.sparse.SparseVec;
import com.winvector.lp.impl.RandomOrder;


/**
 * primal: min c.x: A x = b, x>=0 
 * dual: max y.b: y A <= c 
 * y b = y A x <= c x (by A <=c, x>=0) , so y . b <= c . x at optimal y.b = c.x
 */
public final class LPEQProb implements Serializable, LPEQProbI {
	private static final long serialVersionUID = 1L;
	
	
	public final PreMatrixI A;
	public final double[] b;
	public final PreVecI c;
	

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
	public LPEQProb(final PreMatrixI A_in, final double[] b_in, final PreVecI c_in)
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
	private static void checkParams(final PreMatrixI A_, final double[] b_, final PreVecI c_)
			throws LPException.LPMalformedException {
		if ((A_ == null) || (b_ == null) || (c_ == null)
				|| (A_.rows() != b_.length) || (A_.cols() != c_.dim())) {
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
					if (A_.cols() != c_.dim()) {
						problem = problem + " A_.cols()(" + A_.cols()
								+ ")!=c.length(" + c_.dim() + ")";
					}
				}
			}
			throw new LPException.LPMalformedException(problem);
		}
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
	@Override
	public void checkPrimFeas(final HVec x,
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
	@Override
	public void checkDualFeas(final double[] y,
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
	public void checkPrimDualFeas(final HVec x, final double[] y, double tol) throws LPException {
		if ((tol <= 0.0)||Double.isNaN(tol)||Double.isInfinite(tol)) { 
			tol = 0.0;
		}
		checkPrimFeas(x, tol);
		checkDualFeas(y, tol);
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
	public void checkPrimDualOpt(final HVec x,
			final double[] y, double tol) throws LPException {
		if ((tol <= 0.0)||Double.isNaN(tol)||Double.isInfinite(tol)) { 
			tol = 0.0;
		}
		checkPrimFeas(x, tol);
		checkDualFeas(y, tol);
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
	
	@Override
	public <T extends Matrix<T>> HVec primalSoln(final int[] colBasis, final LinalgFactory<T> factory)
			throws LPException {
		final Matrix<T> AP = factory.matrixCopy(extractColumns(colBasis));
		final double[] xp = AP.solve(b());
		if (xp == null) {
			throw new LPException.LPErrorException("basis solution failed");
		}
		final HVec x = new HVec(colBasis,xp); 
		return x;
	}

	/**
	 * @param p
	 *            primal optimal solution
	 * @param tol
	 *            tolerance for comparisions
	 * @param factory 
	 * @return dual-optimal solution
	 */
	@Override
	public <T extends Matrix<T>> double[] dualSolution(final int[] colBasis, final LinalgFactory<T> factory) throws LPException {
		 // we now have a list of equality constraints to work with
		 final PreMatrixI eqmatT = extractColumns(colBasis);
		 final int nc = colBasis.length;
		 final double[] eqvec = new double[nc];
		 // put in all complementary slackness relns
		 // Schrijver p. 95
		 for (int bi = 0; bi < nc; ++bi) {
			 int i = colBasis[bi];
			 eqvec[bi] = c.get(i);
		 }
		 final T eqmat = factory.matrixCopy(eqmatT.transpose());
		 double[] y = null;
		 if(eqmat.rows()==eqmat.cols()) {
			 try {
				 y = eqmat.solve(eqvec);
			 } catch (Exception ex) {
			 }
		 }
		 if(null==y) {
			 // likely was wrong shape or under rank Ridge form: argmin_y || E y - c ||^2 + epsilon || y ||
			 // which is (E^t E + epsilon I) y = E^t c (yes, I know it degrades condition number)
			 final T eT = factory.matrixCopy(eqmatT);
			 final T eTe = eT.multMat(eqmat);
			 for(int i=0;i<eTe.cols();++i) {
				 eTe.set(i, i, eTe.get(i, i) + 1.0e-7);
			 }
			 final double[] eTc = eT.mult(eqvec);
			 y = eTe.solve(eTc);
		 }
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
		final double[] dualSoln = dualSolution(primSoln.basisColumns,factory);
		checkPrimDualOpt(primSoln.primalSolution, dualSoln, tol);
		return primSoln;
	}
	
	
	public void print(final PrintStream p) {
		p.println();
		p.println("x>=0");
		p.println(A);
		p.print(" * x " + "=" + " ");
		p.println(Matrix.toString(b));
		p.print("minimize x . ");
		p.println(c);
		p.println();
	}
	
	public void print() {
		print(System.out);
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
			for(int j=0;j<c.dim();++j) {
				final double cj = c.get(j);
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
			for(int j=0;j<c.dim();++j) {
				final double aij = A.get(i, j);
				if(Math.abs(aij)!=0) {
					++nnz;
				}
			}
			if(nnz>0) {
				p.print("\teq" + vnf.format(i) + ":\t");
				boolean first = true;
				for(int j=0;j<c.dim();++j) {
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
				p.println(" " + "=" + " " + vvf.format(b[i]));
			}
		}
		p.println();
		p.println("Bounds");
		for(int j=0;j<c.dim();++j) {
			p.println("\t0 <= x" + vnf.format(j));
		}
		p.println();
		p.println("End");
		p.println();
		p.println("\\* eof *\\");
	}

	

	// Abstract LPEQProb methods
	@Override
	public int rows() {
		return A.rows();
	}
	

	@Override
	public SparseVec extractColumn(final int j) {
		return A.extractColumn(j);
	}

	@Override
	public PreMatrixI extractColumns(final int[] basis) {
		return A.extractColumns(basis);
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