package com.winvector.lp.impl;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;


import com.winvector.linagl.LinalgFactory;
import com.winvector.linagl.Matrix;
import com.winvector.linalg.colt.ColtLinAlg;
import com.winvector.linalg.colt.NativeLinAlg;
import com.winvector.lp.LPException;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.TestLP;


public class TestLPImpl {
	public <Z extends Matrix<Z>> void testRank1Update(final LinalgFactory<Z> factory) throws LPException {
		final LPEQProb<Z> prob = TestLP.exampleProblem(factory);
		final RTableau<Z> tab = new RTableau<Z>(prob,new int[] {0,1,2});
		final int leavingI = 0;
		final int enteringI = 3;
		final double[] u = tab.prob.A.extractColumn(enteringI);
		final double[] v = tab.basisSolveRight(u);
		final Z priorBInv = tab.BInv.copy();
		tab.basisPivot(leavingI,enteringI,v);
		final Z incBInv = tab.BInv.copy();
		{
			double maxDiff = 0.0;
			for(int i=0;i<priorBInv.rows();++i) {
				for(int j=0;j<priorBInv.cols();++j) {
					maxDiff = Math.max(maxDiff,Math.abs(priorBInv.get(i,j)-incBInv.get(i,j)));
				}
			}
			assertTrue(maxDiff>0.5);
		}
		tab.resetBasis(tab.basis); // force fresh b inverse calculation
		final Z batchBInv = tab.BInv.copy();
		{
			double maxDiff = 0.0;
			for(int i=0;i<priorBInv.rows();++i) {
				for(int j=0;j<priorBInv.cols();++j) {
					maxDiff = Math.max(maxDiff,Math.abs(batchBInv.get(i,j)-incBInv.get(i,j)));
				}
			}
			assertTrue(maxDiff<1.0e-3);
		}
	}
	
	
	@Test
	public <Z extends Matrix<Z>> void testLPSolverImpl() throws LPException {
		final ArrayList<LinalgFactory<?>> factories = new ArrayList<LinalgFactory<?>>();
		factories.add(NativeLinAlg.factory);
		factories.add(ColtLinAlg.factory);
		for(final LinalgFactory<?> f: factories) {
			testRank1Update(f);
		}
	}
}
