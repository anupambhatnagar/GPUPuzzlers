# GPUPuzzlers

## Repo Layout
```
├── raw                   // a coal mine with diamonds (orignal code snippets by Adnan)
├──
├── Gemfile               // jekyll packages
├── Gemfile.lock          // jekyll packages with pinned dependencies (auto generated)
├── LICENSE               // license file
├── README.md             // this file
├── _config.yml           // main config file for the website
├── _includes             // jekyll include files
├── _layouts              // jekyll layout files
├── _posts                // all posts (rendered by date, latest first)
├── _sass                 // jekyll scss files
├── assets                // scss files
├──
├── index.html            // homepage template
├── tags.html             // tag generation file
├── pages                 // folder containing dedicated pages
├── images                // folder containing images for all pages (not posts)
├──
├── d2h_sync              // content for device to host sync lesson
├── launch_queue          // content on cuda launch queue lesson
└── vector_flops          // content on tflops lesson
```

## Lessons

Name and description - the name should be consistent with the folder name above. The description
should state the topics which will be discussed.

1. vector_flops:  flops and memory bandwith of vector ops and gemm
1. d2h_sync: device to host synchronization and its pitfalls
1. launch_queue: CUDA launch queue
1. pinned_memory: impact of using pinned memory
1. tensor_cores: achieving higher TFLOPS using tensor cores
1. cuda_streams1: concurrent memory transfer and computation using cuda streams
1. cuda_stream2: concurrent communication and computation
1. horizontal_kernel_fusion:
1. vertical_kernel_fusion:
1. memory_access: impact of accessing data from various memory levels
1. nondeterminism:
1. inter_gpu_communication: impact of using nvswitch, pcie
1. nccl_communication:


## Steps to build the website locally (from the main branch)

1. Install ruby `brew install ruby`
1. Install all dependencies (gems) `bundle install`
1. To generate webpages locally execute `jekyll serve --future` from the root of the repo. Go to
   `localhost:4000` to view the website.

Note: The `--future` argument is passed to render future dated posts
