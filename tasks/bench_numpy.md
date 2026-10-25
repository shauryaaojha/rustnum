| benchmark | NumPy median (ms) |
| --- | ---: |
| add (1e6 f64, contiguous) | 3.036 |
| add broadcast ([1000,1000] + [1000]) | 3.040 |
| sum (1e7 f64) | 7.054 |
| mean_axis(0) [1000,1000] | 0.416 |
| transpose().to_contiguous() [1000,1000] | 3.260 |
| exp (1e6 f64) | 5.759 |
| sort (1e6 f64) | 12.020 |
| inv (200x200) | 102.249 |
| svd (100x100) | 6.874 |
| matmul (256x256 f64) | 0.692 |
| matmul (512x512 f64) | 2.683 |
| matmul (1024x1024 f64) | 17.432 |
