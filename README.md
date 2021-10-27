# Waste-Detection

This is a program to detect and classify inorganic waste. The first section detects waste by returning areas with lots of white light through binary thresholding, and areas with a high concentration of edges. Plastic, glass, and metal are generally reflective whereas organic matter tends to not be. By thresholding with a high color value (ie a presence of all colors), then features such as a glare or reflections can be isolated. A high concentration of edges seems to be present in labels on plastic bottles and containers, so these areas are isolated too. The edge image and thresholded image are then combined to create our final binary mask. 


To run the program a fortran compiler needs to be installed. On mac/linux this is fairly straight forward and gfortran can be installed from the commandline. To do this on a mac, Xcode command line tools needs to be installed and is most easily installed with homebrew. On linux it looks something like:

`sudo apt install gfortran`   where apt could be replaced with yum or some other package manager.

On a mac:

`brew install gcc`

gcc comes with gfortran and g++. 

Once this is done, the package can be compiled to a python module with numpy's f2py using:

`f2py -c -m impackage impackage.f90`

If you use a mac (not ARM) then the compiled module is already available.

The main program is run through the bash script program.sh. 

To change the input file, change the variables in the script globals.py, which is a script that only conatins the path for the input file. Other images are abvailbe in the folder called testimages. To add new images simply just put them into this folder. At the moment impackage only supports RGB images. 

Various stages of the threshold mask can be saved and shown in the reduce.py script. They are commented out. It is recommended that this script is run on its own without the bash script if you want to save/see the stages of the binary mask. reduce.py also contains the parameters for the image processing. Since the method is not super robust, it is recommended that these parameters are changed depending on the perspective/lighting/viewpoinnt/camera distance/etc. At the moment this is not ideal but could make for a useful GUI app later...
  
The script NN.py is the neural network used to classify the ROIs. It is not spectacular and more work needs to be done to improve it. 

### Running the program

To run the program after installing and compiling the dependencies, compile and run the bash script program.sh. This script manages the temporary directories/files created to pass between python scripts. 

To compile and run:

`chmod +x program.sh`

`./program.sh`

When the image of classified ROIs apears, press the 0 key to close. Then the user will be prompted with a message asking if they would like to save the ROIs. If the user chooses to save the ROIs they will be prompted to name a new directory and the ROIs will be saved there. The folder created contains the text file of the ROI coordinates and a folder of the images inside each bounding box. The program automatically saves the classified and unclassified images with bounding boxes. 


![boxes2](https://user-images.githubusercontent.com/55775010/138995465-e518c9c9-40ce-487a-8d7f-d141aa23ab94.jpg)
Extracted ROIs (pre-classification)

![boxesClass](https://user-images.githubusercontent.com/55775010/138995540-dad0b001-69eb-4430-a667-97e07a1f18c1.jpg)
Classified ROIs

Ideally the ROIs correctly determined to be false positives would be removed, although at this point that would most likely hinder the waste detection so they are left in. 


