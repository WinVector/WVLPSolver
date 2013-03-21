package com.winvector.lp.impl;

import java.util.Arrays;
import java.util.Random;

import com.winvector.linalg.LinalgFactory;
import com.winvector.linalg.Matrix;
import com.winvector.linalg.sparse.HVec;
import com.winvector.linalg.sparse.SparseVec;
import com.winvector.lp.EarlyExitCondition;
import com.winvector.lp.InspectionOrder;
import com.winvector.lp.LPEQProbI;
import com.winvector.lp.LPException;
import com.winvector.lp.LPException.LPTooManyStepsException;
import com.winvector.lp.LPSoln;

/**
 * basic primal revised simplex method
 * primal: min c.x: A x = b, x>=0
 * based on Strang "Linear Algebra and its Applications" second edition
 * see: http://www.win-vector.com/blog/2012/11/yet-another-java-linear-programming-library/
 * hosted at: https://github.com/WinVector/WVLPSolver
 */
public final class RevisedSimplexSolver extends LPSolverImpl {
	public int debug = 0;
	public double checkTol = 1.0e-8;
	public double enteringTol = 1.0e-5;
	public double leavingTol = 1.0e-7;
	public boolean earlyR = false;                 // allow partial inspection for entering columns
	public boolean resuffle = true;              // re-shuffle inspection order each pass
	private final Random rand = new Random(3252351L);
	// run counters
	public long pivots = 0;
	public long inspections = 0;
	public long totalTimeMS = 0;
	public long inspectionTimeMS = 0;
	public long prePivotTimeMS = 0;
	public long postPivotTimeMS = 0;	

	
	private void endRunTimingUpdate(final long startTimeMS, final long endInspectionMS) {
		final long currentTimeMillis = System.currentTimeMillis();
		totalTimeMS += currentTimeMillis - startTimeMS;
		postPivotTimeMS += currentTimeMillis - endInspectionMS;
	}
	
	public void clearCounters() {
		pivots = 0;
		inspections = 0;
		totalTimeMS = 0;
		inspectionTimeMS = 0;
		prePivotTimeMS = 0;
		postPivotTimeMS = 0;		
	}
	

	
	private <T extends Matrix<T>> void runSimplex(final EnhancedBasis<T> tab, final double tol, 
			final int maxRounds, final EarlyExitCondition earlyExitCondition) throws LPException {
		if (debug > 0) {
			System.out.println("start: " + stringBasis(tab.basis));
		}
		// start timing clear counters
		final long startTimeMS = System.currentTimeMillis();
		final InspectionOrder inspectionOrder = tab.prob.buildOrderTracker(rand);
		final double[] bRatPtr = new double[1];
		final int[] sortedBasis = new int[tab.basis.length];
		double[] b = tab.prob.b();
		int steps = 0;
		while (steps<=maxRounds) {
			final long startRoundMS = System.currentTimeMillis();
			if(null!=earlyExitCondition) {
				if(earlyExitCondition.canExit(tab.basis)) {
					//System.out.println("steps: " + normalSteps + ", inspections: " + inspections + ", ratio: " + (inspections/(double)normalSteps));
					final long endTimeMS = System.currentTimeMillis();
					endRunTimingUpdate(startTimeMS,endTimeMS);
					return;
				}
			}
			++steps;
			++pivots;
			//prob.soln(basis,tol);
			//System.out.println("basis good");
			inspectionOrder.startPass();
			if(resuffle) {
				inspectionOrder.shuffle();
			}
			final double[] lambda = tab.leftBasisSoln();
			final double[] preB = tab.basisSolveRight(b);
			for(int i=0;i<preB.length;++i) { // assume any negative are rounding errors
				preB[i] = Math.max(0.0,preB[i]);
			}
			// find most negative entry of r, if any
			// determines joining variable
			int rEnteringV = -1;
			double bestRi = Double.NaN;
			final long startInspectionMS = System.currentTimeMillis();
			prePivotTimeMS += startInspectionMS-startRoundMS;
			// quick low-object way to check for items in basis
			for(int i=0;i<tab.basis.length;++i) {
				sortedBasis[i] = tab.basis[i];
			}
			Arrays.sort(sortedBasis);
			// try to find pivot
			inspectionLoop:
				while(inspectionOrder.hasNext()) {
					++inspections;
					final int v = inspectionOrder.take(tab.basis,lambda);
					final double ri = tab.computeRI(lambda, v);
					//System.out.println("\t" + v + " ri: " + ri);
					if(ri < -enteringTol) {
						if((rEnteringV < 0)||(ri < bestRi)) {
							final int basisIndex = Arrays.binarySearch(sortedBasis,v);
							if(basisIndex<0) { // not already in basis
								rEnteringV = v;
								bestRi = ri;
								if(earlyR) {
									inspectionOrder.liked(v);
									break inspectionLoop;
								}
							}
						}
					}
					inspectionOrder.disliked(v);
				}
			final long endInspectionMS = System.currentTimeMillis();
			inspectionTimeMS += endInspectionMS - startInspectionMS;
			final int enteringV = rEnteringV;
			if (enteringV < 0) {
				// no entry, at optimum
				endRunTimingUpdate(startTimeMS,endInspectionMS);
				return;
			}
			final SparseVec u = tab.prob.extractColumn(enteringV);
			final double[] binvu = tab.basisSolveRight(u);
			final int leavingI = findLeaving(preB,binvu,bRatPtr);
			if (leavingI < 0) {
				endRunTimingUpdate(startTimeMS,endInspectionMS);
				throw new LPException.LPUnboundedException(
						"problem unbounded");
			}
			if (debug > 0) {
				System.out.print(" leavingI: " + leavingI);
				if (leavingI >= 0) {
					System.out.print(" var=" + tab.basis[leavingI]);
				}
				System.out.println();
			}
			// perform the swap
			tab.basisPivot(leavingI,enteringV,binvu);
			//System.out.println("leave: " + basis[leavingI]);
			final long endRoundMS = System.currentTimeMillis();
			postPivotTimeMS += endRoundMS-endInspectionMS;
		}
		totalTimeMS += System.currentTimeMillis() - startTimeMS;
		throw new LPTooManyStepsException("max steps>" + maxRounds);
	}

	/**
	 * the idea is the preB and preBprime should be non-negative, being solutions from another basis
	 * @param tol >=0
	 * @param preB
	 * @param binvu
	 * @return
	 */
	private <Z extends Matrix<Z>> int findLeaving(final double[] preB, final double[] binvu, final double[] bestRatPointer) {
		// find least ratio of xB[i] to v[i] where v[i]>0
		// determines joining variable
		//(degenerate when xB[i] = 0, could put anti-cycling code here)
		double determiningRat = Double.NaN;
		int leavingI = -1;
		for(int i=0;i<binvu.length;++i) {
			final double vi = binvu[i];
			//System.out.println("l(" + basis[i] + ")= " + vi);
			if (vi>leavingTol) {
				final double xBi = preB[i];
				final double rat = Math.max(0.0,xBi)/vi;
				if ((leavingI<0)
					|| (determiningRat > rat)) {
					determiningRat = rat;
					leavingI = i;
				}
			}
		}
		if(null!=bestRatPointer) {
			bestRatPointer[0] = determiningRat;
		}
		return leavingI;
	}


	
	/**
	 * solve: min c.x: A x = b, x>=0
	 * 
	 * @param prob
	 *            well formed LPProb
	 * @param basis0
	 *            m-vector that is a valid starting basis ( A(basis0) = square
	 *            matrix of basis0 columns x(basis0) = vector with entries
	 *            selected by basis0 then x(basis0) = A(basis0)^-1 b, x>=0 and
	 *            x=0 for non-basis elements) sorted basis0[i+1] > basis0[i]
	 *            allowed to stop if A x = b, x>=0 c.x <=l
	 * @return optimal basis (need not be sorted)
	 * @throws LPException
	 *             (if infeas or unbounded)
	 */
	@Override
	public <T extends Matrix<T>> LPSoln rawSolve(final LPEQProbI prob,
			final int[] basis0, double tol, final int maxRounds, final LinalgFactory<T> factory,
			final EarlyExitCondition earlyExitCondition) throws LPException {
		final long startTimeMS = System.currentTimeMillis();
		if ((tol<=0)||Double.isNaN(tol)||Double.isInfinite(tol)) {
			tol = 0.0;
		}
		final int[] basis;
		{
			final EnhancedBasis<T> t = new EnhancedBasis<T>(prob, basis0,factory);
			runSimplex(t,tol,maxRounds,earlyExitCondition);
			basis = t.basis;
		}
		Arrays.sort(basis); // other t-structures now out of sync with basis, and no longer usable
		final HVec x = prob.primalSoln(basis, factory); // would like to use t.preB, but it isn't correct on all exit conditions and also depens on a sorted basis
		final LPSoln lpSoln = new LPSoln(x, basis, null,0L);
		final long endTimeMS = System.currentTimeMillis();
		lpSoln.reportedRunTimeMS = endTimeMS - startTimeMS; // for our solution only count construction time
		return lpSoln;
	}
}