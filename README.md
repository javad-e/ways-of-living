# A Geography of Indoors for Analyzing Global Ways of Living Using Computer Vision
**Martina Mazzarello, Mikita Klimenka, Rohit Sanatani, Javad Eshtiyagh, Yanhua Yao, Paolo Santi, Fabio Duarte, Richard Florida, Carlo Ratti**

_This paper is currently under review for publication in Nature Cities._

## Abstract
Globalization is often seen as having a homogenizing effect, reducing pronounced local cultural differences. The kinds of housing people live in are one of the most pronounced expressions of local culture. Our research utilizes a unique dataset of more than 640,000 Airbnb images and employs a visual AI framework to examine the differences and similarities in people's living spaces across 80 cities. We show that geographic proximity and socio-economic connectivity—measured by air traffic data—are significantly correlated with the visual features of indoor spaces (R = 0.19 to 0.32), with cities closer to each other or having high passenger traffic between them also exhibiting high visual similarity. Our results challenge the prevailing notion that cultural homogenization may be overdrawn. Despite the homogenizing influence of globalization, distinctive local styles persist in domestic living arrangements. This research highlights the resilience of local identities and cultural distinctions in the face of global pressures.


## Before starting
The code is written in Python 3.9.7. Please use `requirements.txt` file to install all the required packages and dependencies for this project by running:
```
pip install -r requirements.txt
```

## File Descriptions
**run.py -** a script to initiate neural network training <br />
**model_utils.py -** utils function to support neural network training <br />
**run_gradcam.py -** run gradcam analysis on the dataset <br />
**process_results.py -** run trained model on the results and convert them to latent space <br />
