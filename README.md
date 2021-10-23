# Waste-Detection

This is a program to detect and classify inorganic waste. The first section detects waste by returning areas with lots of white light through binary thresholding, and areas with a high concentration of edges. Plastic, glass, and metal are generally reflective whereas organic matter tends to not be. By thresholding with a high color value (ie a presence of all colors), then features such as a glare or reflections can be isolated. A high concentration of edges seems to be present in labels on plastic bottles and containers, so these areas are isolated too. The edge image and thresholded image are then combined to create our final binary mask. 


To run the program a fortran compiler needs to be installed. On mac/linux this is fairly straight forward and gfortran can be installed from the commandline. To do this on a mac, Xcode command line tools needs to be installed and is most easily installed with homebrew. On linux it looks something like:

sudo apt install gfortran

On a mac:

brew install gcc

gcc comes with gfortran and g++. 

Once this is done, the package can be compiled to a python module with numpy's f2py using:

f2py -c -m <module name> impackage.f90

The main program is run through the bash script program.sh. 

To change the input file, change the variables in the script globals.py, which is a script that only conatins the path for the input file. Various stages of the threshold mask can be saved and shown in the reduce.py script. They are commented out. It is recommended that this script is run on its own without the bash script if you want to save/see the stages of the binary mask. reduce.py also contains the parameters for the image processing. Since the method is not super robust, it is recommended that these parameters are changed depending on the perspective/lighting/viewpoinnt/camera distance/etc. At the moment this is not ideal but could make for a useful GUI app later...
  
The script NN.py is the neural network used to classify the ROIs. It is not spectacular and more work needs to be done to improve it. 
