package com.winvector.lp.impl;

import static org.junit.Assert.assertTrue;

import java.util.ArrayList;

import org.junit.Test;

import com.winvector.linagl.LinalgFactory;
import com.winvector.linagl.Matrix;
import com.winvector.linagl.SparseVec;
import com.winvector.linalg.colt.ColtMatrix;
import com.winvector.linalg.colt.NativeMatrix;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPException;
import com.winvector.lp.TestLP;


public class TestLPImpl {
	public <Z extends Matrix<Z>> void testRank1Update(final LinalgFactory<Z> factory) throws LPException {
		final LPEQProb prob = TestLP.exampleProblem(factory);
		final EnhancedBasis<Z> tab = new EnhancedBasis<Z>(prob,new int[] {0,1,2},factory);
		final int leavingI = 0;
		final int enteringI = 3;
		final SparseVec u = tab.prob.extractColumn(enteringI,tab.extractTemps);
		final double[] v = tab.basisSolveRight(u);
		final NativeMatrix priorBInv = tab.BInv.copy(NativeMatrix.factory,false);
		tab.basisPivot(leavingI,enteringI,v);
		final NativeMatrix incBInv = tab.BInv.copy(NativeMatrix.factory,false);
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
		final NativeMatrix batchBInv = tab.BInv.copy(NativeMatrix.factory,false);
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
		factories.add(NativeMatrix.factory);
		factories.add(ColtMatrix.factory);
		for(final LinalgFactory<?> f: factories) {
			testRank1Update(f);
		}
	}
}
