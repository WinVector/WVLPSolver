package com.winvector.lp;

/**
 * Copyright John Mount 2002.  An undisclosed work, all right reserved.
 */

/**
 * LP specific exceptions
 */
public abstract class LPException extends Exception {
	private static final long serialVersionUID = 1L;

	LPException(String s) {
		super(s);
	}

	/**
	 * problem is infeasible
	 */
	public final static class LPInfeasibleException extends LPException {
		private static final long serialVersionUID = 1L;
		public LPInfeasibleException(String s) {
			super(s);
		}
	}

	/**
	 * problem is unbounded
	 */
	public final static class LPUnboundedException extends LPException {
		private static final long serialVersionUID = 1L;
		public LPUnboundedException(String s) {
			super(s);
		}
	}

	/**
	 * problem is malformed
	 */
	public final static class LPMalformedException extends LPException {
		private static final long serialVersionUID = 1L;
		public LPMalformedException(String s) {
			super(s);
		}
	}
	
	/**
	 * top many steps
	 */
	public final static class LPTooManyStepsException extends LPException {
		private static final long serialVersionUID = 1L;
		public LPTooManyStepsException(String s) {
			super(s);
		}
	}

	/**
	 * error in algorithm
	 */
	public final static class LPErrorException extends LPException {
		private static final long serialVersionUID = 1L;
		public LPErrorException(String s) {
			super(s);
		}
	}
}