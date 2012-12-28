package com.winvector.lp.impl;

import java.util.Arrays;


/**
 * comparison key for a basis
 */
final class SBasis implements Comparable<SBasis> {
	public final int[] d;

	public String toString() {
		final StringBuilder b = new StringBuilder();
		b.append('[');
		for (int i = 0; i < d.length; ++i) {
			if (i > 0) {
				b.append(' ');
			}
			b.append(d[i]);
		}
		b.append(']');
		return b.toString();
	}

	public SBasis(final int[] d_in) {
		d = new int[d_in.length];
		for (int i = 0; i < d_in.length; ++i) {
			d[i] = d_in[i];
		}
		if (d.length > 1) {
			Arrays.sort(d);
		}
	}

	public SBasis nb(final int leave, final int join) {
		final int ov = d[leave];
		d[leave] = join;
		final SBasis rb = new SBasis(d);
		d[leave] = ov;
		return rb;
	}

	public int compareTo(final SBasis ob) {
		if(d.length!=ob.d.length) {
			if(d.length>=ob.d.length) {
				return 1;
			} else {
				return -1;
			}
		}
		for (int i = 0; i < d.length; ++i) {
			if (d[i] != ob.d[i]) {
				if (d[i] >= ob.d[i]) {
					return 1;
				} else {
					return -1;
				}
			}
		}
		return 0;
	}

	public boolean equals(final Object o) {
		if (this == o) {
			return true;
		}
		return compareTo((SBasis)o) == 0;
	}

	public int hashCode() {
		int r = 0;
		for (int i = 0; i < d.length; ++i) {
			r += i * d[i];
		}
		return r;
	}
}