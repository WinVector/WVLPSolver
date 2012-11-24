package com.winvector.linalg.colt;

import java.util.Arrays;

import com.winvector.linagl.Vector;

public final class NativeVector extends Vector {
	private static final long serialVersionUID = 1L;
	
	private final double[] underlying;
	
	public NativeVector(final int dim) {
		underlying = new double[dim];
	}
	
	NativeVector(final double[] u) {
		underlying = u;
	}
	
	@Override
	public double get(final int i) {
		return underlying[i];
	}

	@Override
	public void set(final int i, final double v) {
		underlying[i] = v;
	}

	@Override
	public int size() {
		return underlying.length;
	}
	
	@Override
	public NativeVector newVector(final int n) {
		return new NativeVector(n);
	}

	@Override
	public NativeVector copy() {
		return new NativeVector(Arrays.copyOf(underlying,underlying.length));
	}
	
	public static String toString(final double[] v) {
		final StringBuilder b = new StringBuilder();
		b.append("{");
		boolean first = true;
		for(final double vi: v) {
			if(!first) {
				b.append(", ");
			} else {
				first = false;
			}
			b.append("" + vi);
		}
		b.append("}");
		return b.toString();
		
	}
	
	@Override
	public String toString() {
		return toString(underlying);
	}
}
