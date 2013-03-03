package com.winvector.lp.impl;

import java.util.Random;
import java.util.Set;

import com.winvector.lp.InspectionOrder;

public final class RandomOrder implements InspectionOrder {
	
	
	private final Random rand;
	private final int[] ids;
	private int ptr = 0;

	
	public RandomOrder(final int n, final Random rand) {
		this.rand = rand;
		ids = new int[n];
		for(int i=0;i<n;++i) {
			ids[i] = i;
		}
		shuffle();
	}
	
	@Override
	public boolean hasNext() {
		return ptr<ids.length;
	}
	
	@Override
	public int take(final Set<Integer> current, final double[] lambda) {
		final int r = ids[ptr];
		++ptr;
		return r;
	}

	@Override
	public void startPass() {
		ptr = 0;
	}
	
	@Override
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

	@Override
	public void liked(final int v) {
	}

	@Override
	public void disliked(final int v) {
	}
}
