package com.winvector.lp.impl;

import java.util.Random;

public final class InspectionOrder {
	private final int[] perm;
	private int ptr = 0;
	
	public static int[] perm(final Random rand, final int n) {
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

	
	public InspectionOrder(final int n, final Random rand) {
		perm = perm(rand,n);
	}
	
	public int take() {
		final int r = perm[ptr];
		ptr = ptr + 1;
		if(ptr>=perm.length) {
			ptr = 0;
		}
		return r;
	}

	// slow operation
	public void moveToEnd(final int v) {
		// find where v is
		int a = 0;
		while(perm[a]!=v) {
			++a;
		}
		final int end = (ptr<=0)?perm.length-1:ptr-1;
		while(a!=end) { 
			final int next = (a>=perm.length-1)?0:a+1;
			final int vNext = perm[next];
			final int vA = perm[a];
			perm[a] = vNext;
			perm[next] = vA;
			a = next;
		}
	}
}
