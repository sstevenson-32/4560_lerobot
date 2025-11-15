<p>Computer Vision to detect cup location on a grid</p>
<p>Perfect for cup pong tasks! 🤖</p>

<p>
  Stuff left to do:
  1. Parameterize grid size. Also create a bigger grid without corners that are too small. The current bigger grid is too small and therefore noisy and unplayable.
  2. Tweak bounding box for cup base. Currently is fairly accurate, but there is an edge case where the corner of the cup starts on white square, so the bounding box doesn't pick it up on time.
  3. Use extrapolation to check the edges, since currently chessboard corners only checks inner squares. Should just be (corner1 - (corner2 - corner1)) to find where the -1 corner would be.
</p>
