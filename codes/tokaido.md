Tokaido dataset: large-scale synthetic dataset for vision-based structural condition assessment of Japanese high-speed railway viaducts
 
##########################
Please cite:
Narazaki, Y., Hoskere, V., Yoshida, K., Spencer, B. F., & Fujino, Y. (2021). Synthetic environments for vision-based structural condition assessment of Japanese high-speed railway viaducts. Mechanical Systems and Signal Processing, 160, 107850.
 
##########################
images: 1920x1080 png
labels (component, damage): 640x360 bmp (downsampled from the raw 1920x1080 png files)
depth: 640x360 png (downsampled from raw data)
 
files_train.csv: path to synthetic training images (viaducts)
files_test.csv: path to synthetic test images (viaducts)
FORMAT:
image file name, component label file name, damage label file name, depth image file name, camera focal length in mm, regular images, images containing damage in the RRDR (see Narazaki et al. 2021).
 
files_puretex_train.csv: path to synthetic training images (pure texture)
files_puretex_test.csv: path to synthetic test images (pure texture)
FORMAT:
image file name, damage label file name
 
###########################
!!! How to use the data!!!
1. During TRAINING networks for structural component recognition, the participants should use images specified by files_train.csv with column F = True (regular images)
2. During TRAINING networks for damage recognition, the participants should use images specified by files_train.csv with column G = True (images with close-up damage). In addition, the participants can use images specified by files_puretex_train.csv.
3. During TESTING networks for structural component recognition, the participants should to use images specified by files_test.csv with column F = True.
4. During TESTING networks for damage recognition, images specified by files_test.csv with column G = True, as well as images specified by files_puretex_test.csv should be used.
 
#############################
ANNOTATIONS:
1: Structural component recognition
   1 - Nonbridge
   2 - Slab
   3 - Beam
   4 - Column
   5 - Nonstructural components (Poles, Cables, Fences)
   6 - Rail
   7 - Sleeper
   8 - Other components
 
If no label is assigned, the pixel has a value of 0. The pixels with the label 0 will simply be discarded during the evaluation, and therefore have no effect on the team's score.
 
2. Damage recognition
   1. No Damage
   2. Concrete Damage (cracks, spalling)
   3. Exposed Rebar
 
If no label is assigned, the pixel has a value of 0. The pixels with the label 0 will simply be discarded during the evaluation, and therefore have no effect on the team's score. Those "no label" pixels include damage pixels located very far (outside RRDR).
 
3. Depth
Integer values (uint16). Linearly scaled depth between 0.5m and 30m.
0 means 0.5m, and (2**16-1) means 30m. Depth outside this range is clipped to either 0.5m or 30m.