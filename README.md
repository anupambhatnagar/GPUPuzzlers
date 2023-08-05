# GPUPuzzlers

## Puzzlers

1. vector_flops:flops and memory bandwith of vector ops and gemms
1. d2h sync: device to host synchronization and its pitfalls
1. launch queue: cuda launch queue
1. memory: impact of using pinned and pageable memory, cuda caching allocator
1. tensor cores: achieving higher TFLOPS using tensor cores
1. streams: concurrent kernel execution (data transfer and computation using cuda streams)
1. kernel fusion: horizontal and vertical kernel fusion
1. communication: nccl, impact of using nvswitch, pcie

## Repo Layout

```
├── CNAME               // custom domain name
├── Gemfile             // jekyll packages
├── Gemfile.lock        // jekyll packages with pinned dependencies (auto generated)
├── LICENSE             // license file
├── README.domain       // this file
├── _config.yml         // main config file for the website
├── favicon.ico         // browser icon
├── index.html          // homepage template
├── tags.html           // tag generation file
├── _includes           // jekyll include files
├── _layouts            // jekyll layout files
├── _posts              // all posts (rendered by date, most recent first)
├── _sass               // jekyll scss files
├── assets              // more scss files
├── collectives         // puzzler 8
├── d2h_sync            // puzzler 2
├── fusion              // puzzler 7
├── images              // folder containing images for all pages (not posts)
├── launch_queue        // puzzler 3
├── memory              // puzzler 5
├── pages               // folder containing dedicated pages
├── streams             // streams puzzler 6
├── tensor_cores        // puzzler 4
└── vector_flops        // puzzler 1
```
