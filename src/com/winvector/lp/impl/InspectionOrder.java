package com.winvector.lp.impl;

import java.util.Random;

public final class InspectionOrder {
	
	
	private final Random rand;
	private final int[] ids;
	private int ptr = 0;

	
	public InspectionOrder(final int n, final Random rand) {
		this.rand = rand;
		ids = new int[n];
		for(int i=0;i<n;++i) {
			ids[i] = i;
		}
		shuffle();
	}
	
	public boolean hasNext() {
		return ptr<ids.length;
	}
	
	public int take() {
		final int r = ids[ptr];
		++ptr;
		return r;
	}

	public void startPass() {
		ptr = 0;
	}
	
	public void shuffle() {
		startPass();
		for(int i=0;i<ids.length-1;++i) {
			final int j = i + rand.nextInt(ids.length-i);
			final int idi = ids[i];
			final int idj = ids[j];
			ids[i] = idj;
			ids[j] = idi;
		}
	}
}
