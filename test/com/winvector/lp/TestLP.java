package com.winvector.lp;

/**
 * Copyright John Mount, Nina Zumel 2002,2003.  An undisclosed work, all right reserved.
 */



import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;

import org.junit.Test;

import com.winvector.linalg.DenseVec;
import com.winvector.linalg.LinalgFactory;
import com.winvector.linalg.Matrix;
import com.winvector.linalg.PreMatrixI;
import com.winvector.linalg.PreVecI;
import com.winvector.linalg.colt.ColtMatrix;
import com.winvector.linalg.colt.NativeMatrix;
import com.winvector.linalg.sparse.SparseVec;
import com.winvector.lp.LPException.LPMalformedException;
import com.winvector.lp.impl.RevisedSimplexSolver;

/**
 * Junit tests. run with java junit.swingui.TestRunner
 */
public final class TestLP  {
	
	public <Z extends Matrix<Z>> void testLPSolverTrivial(final LinalgFactory<Z> factory) throws LPMalformedException {
		{
			boolean caught1 = false;
			final double[] c = new double[1];
			c[0] = -1.0;
			final LPEQProb prob = new LPEQProb(factory.newMatrix(1, 1,false).columnMatrix(),
					new double[1], new DenseVec(c));
			try {
				final RevisedSimplexSolver solver = new RevisedSimplexSolver();
				solver.solve(prob, null, 0.0, 1000, factory);
			} catch (LPException.LPUnboundedException ue) {
				caught1 = true;
			} catch (LPException le) {
				assertTrue("caught: " + le,false);
			}
			if (!caught1) {
				assertTrue("didn't detect unbounded case",false);
			}
		}
		try {
			final double[] c = new double[1];
			c[0] = 1.0;
			final LPEQProb prob = new LPEQProb(factory.newMatrix(1, 1,false).columnMatrix(),
					new double[1], new DenseVec(c));
			final RevisedSimplexSolver solver = new RevisedSimplexSolver();
			solver.solve(prob, null, 0.0, 1000, factory);
		} catch (LPException le) {
			fail("caught: " + le);
		}
		
	}
	
	public static <Z extends Matrix<Z>> LPEQProb exampleProblem(final LinalgFactory<Z> factory) throws LPException {
		// p. 320 of Strang exercise 8.2.8
		final Matrix<Z> m = factory.newMatrix(3,5,false);
		final double[] b = new double[3];
		final double[] c = new double[5];
		m.set(0,0,1.0); m.set(0,1,1.0); m.set(0,2,-1.0); b[0] = 4.0;   // x1 + x2 - s1 = 4
		m.set(1,0,1.0); m.set(1,1,3.0); m.set(1,3,-1.0); b[1] = 12.0;  // x1 + 3*x2 - s2 = 12
		m.set(2,0,1.0); m.set(2,1,-1.0); m.set(2,4,-1.0);                // x1 - x2 - s3 = 0
		c[0] = 2.0; c[1] = 1.0;                                     // minimize 2*x1 + x2
		final LPEQProb prob = new LPEQProb(m.columnMatrix(),b,new DenseVec(c));
		return prob;
	}
	
	public <Z extends Matrix<Z>> void testLPExample(final LinalgFactory<Z> factory) throws LPException {
		final LPEQProb prob = exampleProblem(factory);
		final LPSoln soln1 = prob.solveDebug(new RevisedSimplexSolver(), 1.0e-6, 1000, factory);
		final double[] expect = {3.00000, 3.00000, 2.00000, 0.00000, 0.00000};
		assertNotNull(soln1);
		assertNotNull(soln1.primalSolution);
		for(int i=0;i<expect.length;++i) {
			assertTrue(Math.abs(soln1.primalSolution.get(i)-expect[i])<1.0e-3);
		}
	}
	

	
	@Test
	public <Z extends Matrix<Z>> void testLPSolverTrivial() throws LPException {
		final ArrayList<LinalgFactory<?>> factories = new ArrayList<LinalgFactory<?>>();
		factories.add(NativeMatrix.factory);
		factories.add(ColtMatrix.factory);
		for(final LinalgFactory<?> f: factories) {
			testLPSolverTrivial(f);
			testLPExample(f);
		}
	}
	
	private static final class LPINEQProb  {
		private final PreMatrixI A;
		private final double[] b;
		private final PreVecI c;

		public LPINEQProb(final PreMatrixI A_in, final double[] b_in, final PreVecI c_in)
				throws LPException.LPMalformedException {
			A = A_in;
			b = b_in;
			c = c_in;
		}
		
		/**
		 * not a good way to convert to an equality problem
		 * @return
		 * @throws LPMalformedException
		 */
		public LPEQProb eqFormSlacks() throws LPMalformedException {
			final int m = A.rows();
			final int n = A.cols();
			final ArrayList<SparseVec> slacks = new ArrayList<SparseVec>(m);
			for(int i=0;i<m;++i) {
				slacks.add(SparseVec.sparseVec(m,i,1.0));
			}
			final double[] newc = new double[n+m];
			for(int j=0;j<n;++j) {
				newc[j] = c.get(j);
			}
			return new LPEQProb(A.addColumns(slacks),b,new DenseVec(newc));
		}
	}

	
	@Test
	public void testShadow() throws LPException {
		final Matrix<?> m = ColtMatrix.factory.newMatrix(4,3,true);
		final double[] b = new double[4];
		final double[] c = new double[3];
		m.set(0,0,1.0); b[0] = 10.0;   // x0 <= 10
		m.set(1,1,1.0); b[1] = 5.0;   // x1 <=  10
		m.set(2,2,1.0); b[2] = 3.0;   // x2 <= 10
		m.set(3,0,1.0); m.set(3,1,1.0); m.set(3,2,1.0); b[3] = 10.0;   // x0 + x1 + x2 <= 10
		c[0] = -10.0; c[1] = -50.0; c[2] = -100.0;                   // maximize 10*x0 + 50*x1 + 100*x2
		final LPEQProb prob = new LPINEQProb(m.columnMatrix(),b,new DenseVec(c)).eqFormSlacks();
		//prob.printCPLEX(System.out);
		final RevisedSimplexSolver solver = new RevisedSimplexSolver();
		final double tol = 1.0e-10;
		final LPSoln soln = solver.solve(prob, null, tol, 1000, ColtMatrix.factory);
		final double[] dual = prob.dualSolution(soln, tol,ColtMatrix.factory);
		assertNotNull(dual);
	}
}