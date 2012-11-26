package com.winvector.lp.impl;

import java.util.BitSet;
import java.util.Random;

import com.winvector.linagl.Matrix;
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
	public double leavingTol = 1.0e-5;
	public double earlyLeavingTol = 1.0e-2;
	public boolean perturb = false;               // perturb b and c by simulated infinitesimals to avoid degenerate cases
	public boolean earlyR = true;                 // allow partial inspection for entering columns
	public Random rand = new Random(3252351L);

	
	private static int[] perm(final Random rand, final int n) {
		final int[] perm = new int[n];
		for(int i=0;i<n;++i) {
			perm[i] = i;
		}
		for(int i=0;i<n-1;++i) {
			final int j = i + rand.nextInt(n-i);
			if(j>i) {
				final int vi = perm[i];
				final int vj = perm[j];
				perm[i] = vj;
				perm[j] = vi;
			}
		}
		return perm;
	}

	
	private <Z extends Matrix<Z>> int[] runSimplex(final RTableau<Z> tab, final double tol, final int maxRounds) throws LPException {
		if (debug > 0) {
			System.out.println("start: " + stringBasis(tab.basis));
		}
		final double[] cPrime;   // optional perturbation
		final double[] bPrime;   // optional perturbation
		if(perturb) {
			// perturb c
			cPrime = new double[tab.prob.c.length];
			for(int i=0;i<cPrime.length;++i) {
				cPrime[i] = rand.nextDouble();  // non-negative entries- don't convert problem to unbounded
			}
			// perturb b, but make sure initial basis is still feasible
			final double[] tmp = new double[tab.prob.c.length];
			for(int i=0;i<tmp.length;++i) {
				tmp[i] = rand.nextDouble();
			}
			bPrime = tab.prob.A.mult(tmp);
		} else {
			cPrime = null;
			bPrime = null;
		}
		final int nvars = tab.prob.A.cols();
		while (tab.normalSteps<=maxRounds) {
			int enteringV = -1;
			{
				//prob.soln(basis,tol);
				//System.out.println("basis good");
				final BitSet curBasisIndicator = new BitSet(nvars);
				for(final int bi: tab.basis) {
					curBasisIndicator.set(bi);
				}
				final double[] lambda = tab.leftBasisSoln(tab.prob.c);
				final double[] lambdaPrime = perturb?tab.leftBasisSoln(cPrime):lambda;
				// find most negative entry of r, if any
				// determines joining variable
				double prevRi = Double.NaN;
				double prevRPi = Double.NaN;
				final int[] perm = earlyR?perm(rand,nvars):null;
				rsearch:
				for(int ii=0;ii<nvars;++ii) {
					final int v = (null!=perm)?perm[ii]:ii;
					if(!curBasisIndicator.get(v)) {
						final double ri = tab.computeRI(lambda, tab.prob.c, v);
						final double rpi = perturb?tab.computeRI(lambdaPrime,cPrime,v):ri;
						if ((ri < -leavingTol) || ((ri <= 0) && (rpi < -leavingTol))) {
							if ((enteringV < 0)
									|| (ri < prevRi)
									|| ((ri <= prevRi) && (rpi < prevRPi)) ) {
								enteringV = v;
								prevRi = ri;
								prevRPi = rpi;
								if(earlyR && (ri<-earlyLeavingTol)) {
									break rsearch;
								}
							}
						}
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
			final double[] preBprime = perturb?tab.basisSolveRight(bPrime):preB;
			if (checkAll) {
				final Matrix<Z> checkMat = tab.prob.A.extractColumns(tab.basis); 
				final double[] check = checkMat.mult(preB);
				final double checkdsq = Matrix.distSq(tab.prob.b,check);
				if (checkdsq > tol*tol) {
					/*
					 * System.out.println("check: " + check);
					 * System.out.println("could not solve: "); Matrix amat =
					 * prob.A.extractColumns(basis);
					 * //amat.print(System.out); Matrix qmat = new
					 * DenseMatrix(QElement.qzero,amat.rows(),amat.cols());
					 * double nativeDen =
					 * amat.zeroElement().newElement(120); double half =
					 * amat.zeroElement().newElement(Factory.newQ(1,2));
					 * ZElement qden = nativeDen.intValue(); for(int i=0;i
					 * <amat.rows();++i) { for(int j=0;j <amat.cols();++j) {
					 * ZElement iv =
					 * amat.get(i,j).mult(nativeDen).add(half).floor();
					 * if(!iv.isZero()) { System.out.println(" {" + i + "," +
					 * j + "," + iv + "," + qden + "},");
					 * qmat.set(i,j,Factory.newQ(iv,qden)); } } }
					 * //qmat.print(System.out); System.out.println("orig: " +
					 * prob.b); Matrix inv = qmat.inverse();
					 * System.out.println("qinvertable: " + (inv!=null));
					 * inv = amat.inverse();
					 * System.out.println("dinvertible: " + (inv!=null));
					 */
					throw new LPErrorException("bad intermediate basis");
				}
			}
			final double[] u = tab.prob.A.extractColumn(enteringV);
			final double[] v = tab.basisSolveRight(u);
			int leavingI = findLeaving(leavingTol, preB, preBprime, v);
			if(leavingI<0) {
				leavingI = findLeaving(0.0, preB, preBprime, v);
			}
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
			tab.basisPivot(leavingI,enteringV,v);
			//System.out.println("leave: " + basis[leavingI]);
		}
		throw new LPTooManyStepsException("max steps>" + maxRounds);
	}

	/**
	 * 
	 * @param tol >=0
	 * @param preB
	 * @param v
	 * @return
	 */
	private <Z extends Matrix<Z>> int findLeaving(final double tol, final double[] preB, final double[] preBprime,
			final double[] v) {
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
		for(int i=0;i<v.length;++i) {
			final double vi = v[i];
			//System.out.println("l(" + basis[i] + ")= " + vi);
			if (vi>tol) {
				final double xBi = preB[i];
				if (xBi>=-checkTol) {
					if (xBi > 0) {
						final double rat = xBi/vi;
						if (Double.isNaN(bestRat)
								|| (bestRat > rat)) {
							bestRat = rat;
							leavingI = i;
						}
					} else if((null!=preBprime)&&(preBprime[i]>0)) {
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
		}
		if(leavingZI>=0) {
			return leavingZI;
		}
		if(leavingPI>=0) {
			return leavingPI;
		}
		if(leavingI>=0) {
			return leavingI;
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
	protected <T extends Matrix<T>> LPSoln<T> rawSolve(final LPEQProb<T> prob, final int[] basis0, double tol, final int maxRounds) 
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
			return new LPSoln<T>(LPEQProb.soln(prob.A, prob.b, rbasis, tol), rbasis);
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