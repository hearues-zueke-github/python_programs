((d) => {
	d.a = 567;
	d.b = "Hello World!";
	console.log(d.b + ", " + d.a);
	console.log(d);

	const fib = (n) => {
		let l = [1, 1];
		let a = 1;
		let b = 1;
		for (let i = 0; i < n; ++i) {
			let c = a + b;
			a = b;
			b = c;
			l.push(b);
		}
		return l;
	};

	const l = fib(6);
	console.log('l: '+l);
})({});
