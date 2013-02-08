package com.winvector.lp.impl;

public interface InspectionOrder {
	void startPass();
	void shuffle();
	boolean hasNext();
	int take();
	void liked(int v);
	void disliked(int v);
}
