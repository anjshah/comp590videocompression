# A1

I implemented a lossless predictive coding strategy with arithmetic coding.For each pixel, I predict its value using the median of the pixel to the left in the current frame, the pixel above in the current frame, and the corresponding pixel in the previous frame. I then encode the residual `(current - prediction) mod 256` using arithmetic coding. To improve compression, I use 256 contexts instead of one global model with the context being determined by spatial activity `(|left - above|)` and temporal disagreement `(|prior - prediction|)`. This method is lossless because the decoder uses the same predictor and reconstructs each pixel exactly from the decoded residual.

Result after testing on bourne.mp4:

Frames encoded: 10
Average size: 2,341,344 bits/frame
Compression ratio: 7.09