package com.winvector.lp.impl;

import java.util.BitSet;
import java.util.Random;

import com.winvector.linagl.Matrix;
import com.winvector.linalg.colt.NativeMatrix;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPException;
import com.winvector.lp.LPException.LPErrorException;
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
	public boolean checkAll = false;
	public double checkTol = 1.0e-8;
	public double enteringTol = 1.0e-5;
	public double earlyEnterTol = 1.0e-2;
	public double leavingTol = 1.0e-5;
	public boolean perturbB = false;               // perturb b by simulated infinitesimals to avoid degenerate cases
	public boolean perturbC = true;               // perturb c by simulated infinitesimals to avoid degenerate cases
	public boolean earlyR = true;                 // allow partial inspection for entering columns
	public Random rand = new Random(3252351L);

	

	
	private <Z extends Matrix<Z>> int[] runSimplex(final RTableau<Z> tab, final double tol, final int maxRounds) throws LPException {
		if (debug > 0) {
			System.out.println("start: " + stringBasis(tab.basis));
		}
		final double[] cPrime;   // optional perturbation
		final double[] bPrime;   // optional perturbation
		if(perturbC) {
			// perturb c
			cPrime = new double[tab.prob.c.length];
			for(int i=0;i<cPrime.length;++i) {
				cPrime[i] = rand.nextDouble();  // non-negative entries- don't convert problem to unbounded
			}
		} else {
			cPrime = null;
		}
		if(perturbB) {
			// perturb b, but make sure initial basis is still feasible
			final double[] tmp = new double[tab.prob.c.length];
			for(int i=0;i<tmp.length;++i) {
				tmp[i] = rand.nextDouble();
			}
			bPrime = tab.prob.A.mult(tmp);
		} else {
			bPrime = null;
		}
		final int nvars = tab.prob.A.cols();
		final int ncond = tab.prob.A.rows();
		final BitSet curBasisIndicator = new BitSet(nvars);
		final InspectionOrder inspectionOrder = new InspectionOrder(nvars,rand);
		while (tab.normalSteps<=maxRounds) {
			int enteringV = -1;
			{
				//prob.soln(basis,tol);
				//System.out.println("basis good");
				inspectionOrder.startPass();
				curBasisIndicator.clear();
				for(final int bi: tab.basis) {
					curBasisIndicator.set(bi);
				}
				final double[] lambda = tab.leftBasisSoln(tab.prob.c);
				final double[] lambdaPrime = perturbC?tab.leftBasisSoln(cPrime):lambda;
				// find most negative entry of r, if any
				// determines joining variable
				double prevRi = Double.NaN;
				double prevRPi = Double.NaN;
				for(int ii=0;ii<nvars;++ii) {
					final int v = inspectionOrder.take();
					if(!curBasisIndicator.get(v)) {
						final double ri = tab.computeRI(lambda, tab.prob.c, v);
						final double rpi = perturbC?tab.computeRI(lambdaPrime,cPrime,v):ri;
						if ((ri < -enteringTol) || ((ri < 0) && (rpi < -enteringTol))) {
							if ((enteringV < 0)
									|| (ri < prevRi)
									|| ((ri <= prevRi) && (rpi < prevRPi)) ) {
								enteringV = v;
								prevRi = ri;
								prevRPi = rpi;
							}
						}
					}
					if(earlyR && (prevRi<-earlyEnterTol) && (ii>2*ncond+5)) {
						break;
					}
				}
				if (debug > 0) {
					System.out.println(" enteringV: " + enteringV + "\t" + prevRi + "\t" + prevRPi);
				}
				if (enteringV < 0) {
					// no entry, at optimum
					return tab.basis();
				}
			}
			final double[] preB = tab.basisSolveRight(tab.prob.b);
			final double[] preBprime = perturbB?tab.basisSolveRight(bPrime):preB;
			if (checkAll) {
				final Matrix<NativeMatrix> checkMat = tab.prob.A.extractColumns(tab.basis,NativeMatrix.factory); 
				final double[] check = checkMat.mult(preB);
				final double checkdsq = Matrix.distSq(tab.prob.b,check);
				if (checkdsq > tol*tol) {
					throw new LPErrorException("bad intermediate basis");
				}
			}
			final double[] u = tab.prob.A.extractColumn(enteringV);
			final double[] binvu = tab.basisSolveRight(u);
			int leavingI = findLeaving(preB, preBprime, binvu);
			if (debug > 0) {
				System.out.print(" leavingI: " + leavingI);
				if (leavingI >= 0) {
					System.out.print(" var=" + tab.basis[leavingI]);
				}
				System.out.println();
			}
			if (leavingI < 0) {
				throw new LPException.LPUnboundedException(
						"problem unbounded");
			}
			// perform the swap
			inspectionOrder.moveToEnd(tab.basis[leavingI]);
			tab.basisPivot(leavingI,enteringV,binvu);
			//System.out.println("leave: " + basis[leavingI]);
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
	private <Z extends Matrix<Z>> int findLeaving(final double[] preB, final double[] preBprime,
			final double[] binvu) {
		// find least ratio of xB[i] to v[i] where v[i]>0
		// determines joining variable
		//(degenerate when xB[i] = 0, could put anti-cycling code here)
		double bestRat = Double.NaN;
		int leavingI = -1;
		// separate record keeping for the xB[i] <=0 case so we can still see relative sizes of the vi
		double bestPRat = Double.NaN;
		int leavingPI = -1;
		// separate record keeping for the xB[i] <=0 case so we can still see relative sizes of the vi
		double bestZRat = Double.NaN;
		int leavingZI = -1;
		for(int i=0;i<binvu.length;++i) {
			final double vi = binvu[i];
			//System.out.println("l(" + basis[i] + ")= " + vi);
			if (vi>leavingTol) {
				final double xBi = preB[i];
				if (xBi>=-leavingTol) {
					final double rat = xBi/vi;
					if (Double.isNaN(bestRat)
							|| (bestRat > rat)) {
						bestRat = rat;
						leavingI = i;
					}
				} else if((null!=preBprime)&&(preBprime[i]>=-leavingTol)) {
					// treat xBi==0 as epsilon, this is where we could have degeneracies and cycle
					final double prat = preBprime[i]/vi;
					if (Double.isNaN(bestPRat)
							|| (bestZRat>prat)) {
						bestPRat = prat;
						leavingPI = i;
					}
				} else {
					// treat xBi==0 as epsilon, this is where we could have degeneracies and cycle
					final double zrat = 1.0/vi;
					if (Double.isNaN(bestZRat)
							|| (bestZRat>zrat)) {
						bestZRat = zrat;
						leavingZI = i;
					}
				}
			}
		}
		if(leavingI>=0) {
			return leavingI;
		}
		if(leavingPI>=0) {
			return leavingPI;
		}
		if(leavingZI>=0) {
			return leavingZI;
		}
		return -1;
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
	protected <T extends Matrix<T>> LPSoln rawSolve(final LPEQProb<T> prob, final int[] basis0, double tol, final int maxRounds) 
			throws LPException {
		if ((tol<=0)||Double.isNaN(tol)||Double.isInfinite(tol)) {
			tol = 0.0;
		}
		final RTableau<T> t = new RTableau<T>(prob, basis0);
		final int[] rbasis = runSimplex(t,tol,maxRounds);
		//System.out.println("steps: " + t.normalSteps);
		if (rbasis == null) {
			return null;
		}
		try {
			return new LPSoln(LPEQProb.soln(prob.A, prob.b, rbasis, tol), rbasis);
		} catch (LPException e) {
			//		    System.out.println("{");
			//		    for(int i=0;i<basis0.length;++i) {
			//			System.out.print(" " + basis0[i]);
			//		    }
			//		    System.out.println("}");
			//		    prob.print(System.out);
			throw e;
		}
	}
}