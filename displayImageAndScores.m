function displayImageAndScores(tl,im,s,t)
% The displayImageAndScoresForNIMA function displays an image (im) in a
% tile of a tiledLayoutl (tl). The title of the tile contains information
% information about the image (t), and scores (s).
%
% Copyright 2020 The MathWorks, Inc.

nexttile(tl);
imshow(im);
title([t; ...
    "Score: "+num2str(s)])

end