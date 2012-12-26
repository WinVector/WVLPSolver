package com.winvector.lp.impl;

import java.util.BitSet;
import java.util.Random;

public final class InspectionOrder {
	
	
	private final Random rand;
	private final int nids;
	private final int[] ids;
	private int ptr = 0;

	
	public InspectionOrder(final int n, final Random rand, final BitSet skips) {
		this.rand = rand;
		int index = 0;
		for(int i=0;i<n;++i) {
			if((null==skips)||(!skips.get(i))) {
				++index;
			}
		}
		nids = index;
		ids = new int[nids];
		index = 0;
		for(int i=0;i<n;++i) {
			if((null==skips)||(!skips.get(i))) {
				ids[index] = i;
				++index;
			}
		}
		shuffle();
	}
	
	public boolean hasNext() {
		return ptr<nids;
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
		for(int i=0;i<nids-1;++i) {
			final int j = i + rand.nextInt(nids-i);
			final int idi = ids[i];
			final int idj = ids[j];
			ids[i] = idj;
			ids[j] = idi;
		}
	}
}
