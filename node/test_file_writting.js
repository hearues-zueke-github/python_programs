const fs = require('fs');

function main() {
	fs.writeFile("/tmp/test", "Hey there!", function(err) {
		if (err) {
			return console.log(err);
		}
		console.log("The file was saved!");
	});

	// Or
	fs.writeFileSync('/tmp/test-sync', 'Hey there2!');
}

if (require.main === module) {
	main();
}
