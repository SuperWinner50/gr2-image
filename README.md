## gr2-image
Gr2-image is a program to convert images into a radar-readable format, specifically tuned for GR2 (Gibson Ridge 2).
To view usage, run the program with `--help` to view the entire help message.
## Features:
* Unweighted antialiasing algorithm
* Kmeans algorithm to fit a colormap to the image, or just using minimal RGB colormap (6 colors of RGB each)
* To smooth out the generating colormap, the following algorithms can be used (other than none):
	* Simulated annealing
	* TSP (Traveling Salesman Problem) heuristic solver (LKH) to find perfect or near perfect solutions (Recommended)
* Variable radar name, scale of image, radar placement in image, radar resolution
## How to use
* Install Rust
* Clone the repository to a local folder, and navigate to it
* Run `cargo run -r -- --help` to view the help menu. To run with options, replace `--help` with what you want.
* Running directly from a binary is also possible