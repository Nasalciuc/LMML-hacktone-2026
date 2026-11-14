**SIGMOID_UNREAD**

The injected rows look normal one at a time. What gives them away is that price, availability and reviews_per_month drift smoothly with map position.
In the 5-d space (lon, lat, log price, availability, log reviews/month), the distance to the 3rd nearest neighbour is bimodal. DBSCAN on the low mode gives six flat groups (1,967 rows), one per letter.
Each letter is a 2-d glyph mapped linearly into the five columns. Whitened PCA plus a rotation-only fit against A-Z skeletons reads them as U, N, R, E, A, D.
The order is not encoded in the data (west to east they read R E N A D U), so the word is the only English word these six letters make.
Close calls: N/Z (same shape turned 90 degrees), D/O (fits almost tied), U/C. Alternatives along those calls: DANCER, CANOER.
Files: `point_par_point.ipynb`, `word.png` plus one `letter_*.png` per letter, `labels.csv` (listing_id, cluster, letter, position).
