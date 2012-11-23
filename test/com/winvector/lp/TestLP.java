package com.winvector.lp;

/**
 * Copyright John Mount, Nina Zumel 2002,2003.  An undisclosed work, all right reserved.
 */



import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;

import org.junit.Test;

import com.winvector.linagl.LinalgFactory;
import com.winvector.linagl.Matrix;
import com.winvector.linagl.Vector;
import com.winvector.linalg.colt.ColtLinAlg;
import com.winvector.linalg.colt.NativeLinAlg;
import com.winvector.lp.impl.RevisedSimplexSolver;

/**
 * Junit tests. run with java junit.swingui.TestRunner
 */
public final class TestLP  {
	
	public <Z extends Matrix<Z>> void testLPSolverTrivial(final LinalgFactory<Z> factory) {
		boolean caught1 = false;
		try {
			final Vector c = factory.newVector(1);
			c.set(0, -1.0);
			final LPEQProb<Z> prob = new LPEQProb<Z>(factory.newMatrix(1, 1,false),
					factory.newVector(1), c);
			final RevisedSimplexSolver<Z> solver = new RevisedSimplexSolver<Z>();
			solver.solve(prob, null, 0.0,1000);
		} catch (LPException.LPUnboundedException ue) {
			caught1 = true;
		} catch (LPException le) {
			assertTrue("caught: " + le,false);
		}
		if (!caught1) {
			assertTrue("didn't detect unbounded case",false);
		}
		try {
			final Vector c = factory.newVector(1);
			c.set(0, 1.0);
			final LPEQProb<Z> prob = new LPEQProb<Z>(factory.newMatrix(1, 1,false),
					factory.newVector(1), c);
			final RevisedSimplexSolver<Z> solver = new RevisedSimplexSolver<Z>();
			solver.solve(prob, null, 0.0,1000);
		} catch (LPException le) {
			fail("caught: " + le);
		}
		
	}
	
	public static <Z extends Matrix<Z>> LPEQProb<Z> exampleProblem(final LinalgFactory<Z> factory) throws LPException {
		// p. 320 of Strang exercise 8.2.8
		final Matrix<Z> m = factory.newMatrix(3,5,false);
		final Vector b = factory.newVector(3);
		final Vector c = factory.newVector(5);
		m.set(0,0,1.0); m.set(0,1,1.0); m.set(0,2,-1.0); b.set(0,4.0);   // x1 + x2 - s1 = 4
		m.set(1,0,1.0); m.set(1,1,3.0); m.set(1,3,-1.0); b.set(1,12.0);  // x1 + 3*x2 - s2 = 12
		m.set(2,0,1.0); m.set(2,1,-1.0); m.set(2,4,-1.0);                // x1 - x2 - s3 = 0
		c.set(0,2.0); c.set(1,1.0);                                     // minimize 2*x1 + x2
		final LPEQProb<Z> prob = new LPEQProb<Z>(m,b,c);
		return prob;
	}
	
	public <Z extends Matrix<Z>> void testLPExample(final LinalgFactory<Z> factory) throws LPException {
		final LPEQProb<Z> prob = exampleProblem(factory);
		final LPSoln<Z> soln1 = prob.solveDebugByInspect(new RevisedSimplexSolver<Z>(), 1.0e-6,factory,1000);
		final double[] expect = {3.00000, 3.00000, 2.00000, 0.00000, 0.00000};
		assertNotNull(soln1);
		assertNotNull(soln1.x);
		assertEquals(expect.length,soln1.x.size());
		for(int i=0;i<expect.length;++i) {
			assertTrue(Math.abs(soln1.x.get(i)-expect[i])<1.0e-3);
		}
	}
	

	
	@Test
	public <Z extends Matrix<Z>> void testLPSolverTrivial() throws LPException {
		final ArrayList<LinalgFactory<?>> factories = new ArrayList<LinalgFactory<?>>();
		factories.add(NativeLinAlg.factory);
		factories.add(ColtLinAlg.factory);
		for(final LinalgFactory<?> f: factories) {
			testLPSolverTrivial(f);
			testLPExample(f);
		}
	}

}