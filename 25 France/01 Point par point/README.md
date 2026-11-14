**SIGMOID_CORNEA**

The injected rows look normal one at a time. What gives them away is that their price, availability and reviews_per_month change smoothly with their position on the map.
In the 5-d space (lon, lat, log price, availability, log reviews/month), the distance to the 3rd nearest neighbour is bimodal. 1,790 rows fall in the low mode, and DBSCAN splits them into six groups (1,685 dots; the rest are noise).
Each group is flat (2 PCs explain about 95% of its variance), so each one is a letter drawn in 2-d and mapped linearly into the five columns. Whitened PCA plus a rotation or mirror makes it readable: C, O, R, N, E, A.
No column orders the groups, so the order comes from the only word the six letters make: CORNEA.
Files: `point_par_point.ipynb` (code), `word.png` plus one `letter_*.png` per letter, and `labels.csv` (listing_id, cluster, letter, position).
