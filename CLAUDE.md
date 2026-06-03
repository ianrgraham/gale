This is a project to experiment with GPU programming for useful physics simulations within native Rust code.

Back at the end of my computational physics PhD in soft matter, I did some research on viscoelastic fluid flows. For this task I used OpenFOAM with rheoTool, but I found these tools to be quite archaic and lacking in many ways. At this same time in my studies I also learned about some ways to run fluid simulations on the GPU, particularly I heard about discontinuous Galerkin methods. At the time, I didn't have much time to explore these, or make an proof of concept implementation. Now that we have fairly good AI tools (looking at you Claude), I'd like to give this a shot.

I'm also a big fan of Rust. Back in my PhD, I did a lot of work in C++/CUDA, and while by the end of it I had a good handle on programming high-throughput simulations in C++, it's not my favorite language. Around the same time I also found rust, and used it in a lot of my CPU-based analysis code, often using pyo3 to bridge them to other Python analysis code I had. But recently released was the [cuda oxide](https://github.com/NVlabs/cuda-oxide) project, which does the best job that I've yet seen in supporting CUDA device kernels in native rust code.

I've add the basic scaffold of a cuda-oxide project, which may be executed from the terminal with `cargo oxide run`, or built with `cargo oxide build`.

While we'll focus initially on Rust-only usage is fine, I'd like for this to eventually have a Python entry point as well, using pyo3 as the interface. 

Multi-GPU is a must here. I myself am running two Titan Vs on a 64 core EPYC 7702P machine with 256GB of memory. We likely want to maintain flexibility between CPU & GPU dispatch for various parts of the simulation workload, depending upon what it is. I also want to make sure we have 1st class support of the immersed boundary method with our DG method sim. 