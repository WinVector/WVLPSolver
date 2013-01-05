package com.winvector.lp.impl;

import java.util.BitSet;
import java.util.Random;

import com.winvector.linagl.LinalgFactory;
import com.winvector.linagl.Matrix;
import com.winvector.lp.EarlyExitCondition;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPException;
import com.winvector.lp.LPException.LPTooManyStepsException;
import com.winvector.lp.LPSoln;
import com.winvector.sparse.SparseVec;

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
	public boolean perturbC = true;
	public boolean perturbB = false;               // this does change which bases are co-incident and which are feasible
	public boolean earlyR = true;                 // allow partial inspection for entering columns
	public boolean earlyLeavingCalc = false;       // value and sort steps early
	public boolean earlyLeavingExit = false;       // allow early inspection exit on valuation calc
	public boolean resuffle = false;               // re-shuffle inspection order each pass
	private final Random rand = new Random(3252351L);

	

	
	private <T extends Matrix<T>> int[] runSimplex(final RTableau<T> tab, final double tol, 
			final int maxRounds, final EarlyExitCondition earlyExitCondition) throws LPException {
		if (debug > 0) {
			System.out.println("start: " + stringBasis(tab.basis));
		}
		final int nvars = tab.prob.A.cols;
		final BitSet curBasisIndicator = new BitSet(nvars);
		for(final int bi: tab.basis) {
			curBasisIndicator.set(bi);
		}
		final InspectionOrder inspectionOrder = new InspectionOrder(nvars,rand);
		final double[] bRatPtr = new double[1];
		double[] c = tab.prob.c;
		double[] b = tab.prob.b;
		if(perturbC) {
			c = new double[tab.prob.c.length];
			for(int i=0;i<c.length;++i) {
				c[i] = tab.prob.c[i]*(1+1.0e-5*rand.nextGaussian()) + 1.0e-7*rand.nextGaussian();
			}
		}
		if(perturbB) {
			b = new double[tab.prob.b.length];
			final double[] x = new double[tab.prob.c.length];
			for(int i=0;i<x.length;++i) {
				x[i] = 1.0e-7*rand.nextDouble(); // keep initial basis feasible
			}
			final double[] p = tab.prob.A.mult(x);
			for(int i=0;i<b.length;++i) {
				b[i] = tab.prob.b[i] + p[i];
			}
		}
		while (tab.normalSteps<=maxRounds) {
			//prob.soln(basis,tol);
			//System.out.println("basis good");
			inspectionOrder.startPass();
			if(resuffle) {
				inspectionOrder.shuffle();
			}
			curBasisIndicator.clear();
			for(final int bi: tab.basis) {
				curBasisIndicator.set(bi);
			}
			final double[] lambda = tab.leftBasisSoln(c);
			final double[] preB = tab.basisSolveRight(b);
			for(int i=0;i<preB.length;++i) { // assume any negative are rounding errors
				preB[i] = Math.max(0.0,preB[i]);
			}
			// find most negative entry of r, if any
			// determines joining variable
			int rEnteringV = -1;
			double bestRi = Double.NaN;
			// step valued version of inspection
			double bestStepValue = Double.NaN;
			int leavingI = -1;
			int matchingEnterV = -1;
			double[] bestBinvu = null;
			inspectionLoop:
			while(inspectionOrder.hasNext()) {
				++tab.inspections;
				final int v = inspectionOrder.take();
				if(!curBasisIndicator.get(v)) {
					final double ri = tab.computeRI(lambda, c, v);
					//System.out.println("\t" + v + " ri: " + ri);
					if(ri < -enteringTol) {
						if((rEnteringV < 0)||(ri < bestRi)) {
							rEnteringV = v;
							bestRi = ri;
							if(earlyR) {
								break inspectionLoop;
							}
						}
						if(earlyLeavingCalc) {
							final SparseVec u = tab.prob.A.extractColumn(rEnteringV);
							final double[] binvu = tab.basisSolveRight(u);
							int leavingIndex = findLeaving(preB,binvu,bRatPtr);
							if((leavingIndex>=0)&&(bRatPtr[0]>=0)) {
								final double stepValue = -ri*bRatPtr[0];
								if(stepValue>=0) {
									if((leavingI<0)||(stepValue>bestStepValue)) {
										leavingI = leavingIndex;
										matchingEnterV = v;
										bestStepValue = stepValue;
										bestBinvu = binvu;
										if(earlyLeavingExit && (bestStepValue>0)) {
											break inspectionLoop;
										}
									}
								}
							}
						}
					}
				}
			}
			if (debug > 0) {
				System.out.println(" rEnteringV: " + rEnteringV + "\t" + bestRi 
						+ ",\tleavingI:\t" + leavingI + "\t" + leavingI + "\t" + matchingEnterV + "\t" + bestStepValue);
			}
			final int enteringV;
			if(leavingI>=0) {
				enteringV = matchingEnterV;
			} else {
				enteringV = rEnteringV;
				if(enteringV>=0) {
					final SparseVec u = tab.prob.A.extractColumn(enteringV);
					final double[] binvu = tab.basisSolveRight(u);
					leavingI = findLeaving(preB,binvu,bRatPtr);
					if(leavingI>=0) {
						bestBinvu = binvu;
						bestStepValue = -bestRi*bRatPtr[0];
					}
				}
			}
			if (debug > 0) {
				System.out.print(" leavingI: " + leavingI);
				if (leavingI >= 0) {
					System.out.print(" var=" + tab.basis[leavingI]);
				}
				System.out.println();
			}
			if (enteringV < 0) {
				// no entry, at optimum
				return tab.basis();
			}
			if (leavingI < 0) {
				throw new LPException.LPUnboundedException(
						"problem unbounded");
			}
			// perform the swap
			tab.basisPivot(leavingI,enteringV,bestBinvu);
			//System.out.println("leave: " + basis[leavingI]);
			if(null!=earlyExitCondition) {
				if(earlyExitCondition.canExit(tab.basis)) {
					return tab.basis();
				}
			}
		}
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
	protected <T extends Matrix<T>> LPSoln rawSolve(final LPEQProb prob,
			final int[] basis0, double tol, final int maxRounds, final LinalgFactory<T> factory,
			final EarlyExitCondition earlyExitCondition) throws LPException {
		if ((tol<=0)||Double.isNaN(tol)||Double.isInfinite(tol)) {
			tol = 0.0;
		}
		final RTableau<T> t = new RTableau<T>(prob, basis0,factory);
		final int[] rbasis = runSimplex(t,tol,maxRounds,earlyExitCondition);
		//System.out.println("steps: " + t.normalSteps);
		//System.out.println("" + "nvars" + "\t" + "ncond" + "\t" + "steps" + "\t" + "inspections");
		//System.out.println("" + prob.nvars() + "\t" + prob.b.length + "\t" + t.normalSteps + "\t" + t.inspections);
		if (rbasis == null) {
			return null;
		}
		final LPSoln lpSoln = new LPSoln(LPEQProb.primalSoln(prob.A, prob.b, rbasis, tol, factory), rbasis);
		lpSoln.stepsTaken = t.normalSteps;
		return lpSoln;
	}
}