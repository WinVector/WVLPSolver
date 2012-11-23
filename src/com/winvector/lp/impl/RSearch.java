package com.winvector.lp.impl;

import java.util.SortedSet;
import java.util.TreeSet;

import com.winvector.linagl.Matrix;
import com.winvector.linagl.Vector;
import com.winvector.lp.LPEQProb;

/**
 * brute force search for a better basis
 */
final class RSearch<Z extends Matrix<Z>> {
	private final LPEQProb<Z> p;

	private final Vector cPrime;

	private final double tol;

	private final SortedSet<SBasis> pending = new TreeSet<SBasis>();
	
	private final SortedSet<SBasis> saw = new TreeSet<SBasis>();
	
	public int escapeSteps = 0;


	public RSearch(final LPEQProb<Z> p_in, final Vector cPrime_in, final double tol_in) {
		p = p_in;
		cPrime = cPrime_in;
		tol = tol_in;
	}

	public SBasis take() {
		if (pending.isEmpty()) {
			return null;
		}
		SBasis r = (SBasis) pending.first();
		pending.remove(r);
		return r;
	}

	private double value(final SBasis b, final Vector obj, final double tol) {
		return b.value(p, obj, tol);
	}

	public SBasis escape(SBasis b) {
		final double vR = value(b, p.c, tol);
		final double vP = value(b, cPrime, tol);
		pending.add(b);
		while (!pending.isEmpty()) {
			b = take();
			if (saw.contains(b)) {
				continue;
			}
			saw.add(b);
			int[] cbasis = RevisedSimplexSolver.complementaryColumns(p.A.cols(), b.d);
			for (int i = 0; i < b.d.length; ++i) {
				for (int j = 0; j < cbasis.length; ++j) {
					final SBasis trial = b.nb(i, cbasis[j]);
					if ((!pending.contains(trial) && (!saw
							.contains(trial)))) {
						final double vbR = value(trial, p.c, tol);
						final double vbP = value(trial, cPrime, tol);
						boolean worse = true;
						boolean better = false;
						if (!Double.isNaN(vbR)) {
							final int rCmp = Double.compare(vbR,vR);
							if (rCmp <= 0) {
								if (rCmp < 0) {
									worse = false;
									better = true;
								} else {
									final int pCmp = Double.compare(vbP,vP);
									if (pCmp <= 0) {
										worse = false;
										if (pCmp < 0) {
											better = true;
										}
									}
								}
							}
						}
						++escapeSteps;
						if (Double.isNaN(vbR) || worse) {
							// not basis or worse objective value
							saw.add(trial);
							pending.remove(trial);
						} else if (better) {
							// found real improvement, all done
							return trial;
						} else {
							// schedule for expansion
							pending.add(trial);
						}
					}
				}
			}
		}
		return null; // never could improve
	}
}