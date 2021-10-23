# Waste-Detection

This is a program to detect and classify inorganic waste. The first section detects waste by returning areas with lots of white light through binary thresholding, and areas with a high concentration of edges. Plastic, glass, and metal are generally reflective whereas organic matter tends to not be. By thresholding with a high color value (ie a presence of all colors), then features such as a glare or reflections can be isolated. A high concentration of edges seems to be present in labels on plastic bottles and containers, so these areas are isolated too. The edge image and thresholded image are then combined to create our final binary mask. 


