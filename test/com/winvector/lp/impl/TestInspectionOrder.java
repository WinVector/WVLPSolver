package com.winvector.lp.impl;


import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.junit.Test;

public class TestInspectionOrder {
	@Test
	public void testShift() {
		final Random rand = new Random(3626L);
		final InspectionOrder io = new InspectionOrder(5,rand,null);
		io.startPass();
		io.take();
		io.take();
		io.moveToEnd(3);
		io.startPass();
		for(int i=0;i<4;++i) {
			assertTrue(io.take()!=3);
		}
		assertTrue(io.take()==3);
	}
}