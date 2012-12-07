package com.winvector.lp;

import com.winvector.linagl.LinalgFactory;
import com.winvector.linagl.Matrix;



/**
 * Copyright John Mount 2002.  An undisclosed work, all right reserved.
 */

/**
 * primal: min c.x: A x = b, x>=0 dual: max y.b: y A <= c y b = y A x <= c x (by
 * y A <=c, x>=0) , so y . b <= c . x at optimal y.b = c.x
 */
public interface LPSolver {
	/**
	 * @param prob
	 *            well formed LPProb
	 * @param basis_in
	 *            (optional) valid initial basis
	 * @return x n-vector s.t. A x = b and x>=0 and c.x minimized allowed to
	 *         stop if A x = b, x>=0 c.x <=l
	 * @throws LPException
	 *             (if infeas or unbounded)
	 */
	<T extends Matrix<T>> LPSoln solve(LPEQProb prob, int[] basis_in, double tol, final int maxRounds, LinalgFactory<T> factory)
			throws LPException;
}