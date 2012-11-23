package com.winvector.lp.impl;

import java.util.BitSet;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;



import com.winvector.linagl.Matrix;
import com.winvector.linagl.Vector;
import com.winvector.lp.LPException;
import com.winvector.lp.LPException.LPErrorException;
import com.winvector.lp.LPException.LPTooManyStepsException;
import com.winvector.lp.LPEQProb;
import com.winvector.lp.LPSoln;

/**
 * basic primal simplex method
 */
public final class RevisedSimplexSolver<T extends Matrix<T>> extends LPSolverImpl<T> {
	public int debug = 0;
	public boolean checkAll = false;
	public double leavingTol = 1.0e-5;
	public double earlyLeavingTol = 1.0e-2;
	public boolean breakCycles = false;
	public boolean perturb = false;
	public boolean earlyR = true;
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
		final Set<SBasis> recentBases;
		if(breakCycles) {
			recentBases = new TreeSet<SBasis>();
		} else {
			recentBases = null;
		}
		final int nvars = tab.prob.A.cols();
		double lastVal = Double.NaN;
		while (tab.normalSteps<=maxRounds) {
			int enteringV = -1;
			{
				//prob.soln(basis,tol);
				//System.out.println("basis good");
				final BitSet curBasisIndicator = new BitSet(nvars);
				for(final int bi: tab.basis) {
					curBasisIndicator.set(bi);
				}
				final Vector lambda = tab.leftBasisSoln(tab.prob.c);
				final Vector lambdaPrime = perturb?tab.leftBasisSoln(tab.cPrime):lambda;
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
						final double rpi = perturb?tab.computeRI(lambdaPrime,tab.cPrime, v):ri;
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
			final Vector preB = tab.basisSolveRight(tab.prob.b);
			if (checkAll) {
				final Matrix<Z> checkMat = tab.prob.A.extractColumns(tab.basis); 
				final Vector check = checkMat.mult(preB);
				final double checkdsq = tab.prob.b.distSq(check);
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
			final Vector u = tab.prob.A.extractColumn(enteringV);
			final Vector v = tab.basisSolveRight(u);
			int leavingI = findLeaving(leavingTol, preB, v);
			if(leavingI<0) {
				leavingI = findLeaving(0.0, preB, v);
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
			// record this basis and check for cycling
			if(null!=recentBases) {
				final SBasis sstart = new SBasis(tab.basis);
				if (recentBases.contains(sstart)) {
					// found cycle
					// take brute-force search until we escape
					// really should never get here as we have the infinitesimal object perturbation in to fight cycles
					recentBases.clear();
					final RSearch<Z> searcher = new RSearch<Z>(tab.prob, tab.cPrime, tol);
					//System.out.println("start cycle search");
					final SBasis send = searcher.escape(sstart);
					if (send == null) {
						// no more improvements, at optimal
						return tab.basis();
					}
					//System.out.println("done cycle search");
					// force step
					tab.resetBasis(send.d);
					recentBases.add(send);
					continue;
				} else {
					if(null!=recentBases) {
						recentBases.add(sstart);
					}
				}
				final double objVal = tab.prob.c.extract(tab.basis).dot(preB);
				if ( Double.isNaN(lastVal) || (lastVal>objVal) ) {
					// 	non-trivial improvement
					lastVal = objVal;
					recentBases.clear();
				}
			}
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
	private static <Z extends Matrix<Z>> int findLeaving(final double tol, final Vector preB,
			final Vector v) {
		// find least ratio of xB[i] to v[i] where v[i]>0
		// determines joining variable
		//(degenerate when xB[i] = 0, could put anti-cycling code here)
		double bestRat = Double.NaN;
		int leavingI = -1;
		// separate record keeping for the xB[i] <=0 case so we can still see relative sizes of the vi
		double bestZRat = Double.NaN;
		int leavingZI = -1;
		{
			int i = -1;
			while ((i = v.nextIndex(i)) >= 0) {
				final double vi = v.get(i);
				//System.out.println("l(" + basis[i] + ")= " + vi);
				if (vi>tol) {
					final double xBi = preB.get(i);
					if (xBi > 0) {
						final double rat = xBi/vi;
						if (Double.isNaN(bestRat)
								|| (bestRat > rat)) {
							bestRat = rat;
							leavingI = i;
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
		if (leavingZI >= 0) { // any xBi<=0 solution dominates all xBi>= solutions
			// use one of the zero ratios
			leavingI = leavingZI;
			bestRat = bestZRat;
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
	protected LPSoln<T> rawSolve(final LPEQProb<T> prob, final int[] basis0, double tol, final int maxRounds) 
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