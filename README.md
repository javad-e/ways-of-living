# A Geography of Indoors for Analyzing Global Ways of Living Using Computer Vision
**Martina Mazzarello, Mikita Klimenka, Rohit Sanatani, Javad Eshtiyagh, Yanhua Yao, Paolo Santi, Fabio Duarte, Richard Florida, Carlo Ratti**

_This paper is currently under review for publication in Nature Cities (Manuscript #: NATCITIES-24070753)._

## Abstract
Globalization is often seen as having a homogenizing effect, reducing pronounced local cultural differences. The kinds of housing people live in are one of the most pronounced expressions of local culture. Our research utilizes a unique dataset of more than 640,000 Airbnb images and employs a visual AI framework to examine the differences and similarities in people's living spaces across 80 cities. We show that geographic proximity and socio-economic connectivity—measured by air traffic data—are significantly correlated with the visual features of indoor spaces (R = 0.19 to 0.32), with cities closer to each other or having high passenger traffic between them also exhibiting high visual similarity. Our results challenge the prevailing notion that cultural homogenization may be overdrawn. Despite the homogenizing influence of globalization, distinctive local styles persist in domestic living arrangements. This research highlights the resilience of local identities and cultural distinctions in the face of global pressures.


## Before starting
The code is written in Python 3.9.7. Please use `requirements.txt` file to install all the required packages and dependencies for this project by running:
```
pip install -r requirements.txt
```
Typical install time: 15 mins

## File Descriptions
**run.py -** a script to initiate neural network training. Average run time:  <br />
**model_utils.py -** utils function to support neural network training. <br />
**run_gradcam.py -** run gradcam analysis on the dataset. Average run time:  <br />
**process_results.py -** run trained model on the results and convert them to latent space. Average run time:  <br />
**flight_analysis.ipynb -** analyzes geographic and flight data and correlation with image similarity. Average run time: ~2 minute per room type <br />
**test_metrics_external_dataset.ipynb -** analyzes the representativeness of Airbnb data with regards to other datasets. Average run time: ~2 minute per room type <br />

## General Instructions and Workflow

The general workflow for our analysis is as follows:

- Train neural network on Airbnb data using city names as target labels (run.py). 
- Process the outputs of the model on test data and generate a latent space representation (process_results.py)
- Analyze correlations between geographic distance and latent space distance across all city pairs.
- Analyze correlations between flight traffic and visual similarity (flight_analysis.ipynb). 
- Analyze correlations between geographic distance and KOF globalization index across all city pairs. 
- Visualise model activations and identify semantic parts of images that are important visual identifiers of cities. 
- Analyze the representativeness of Airbnb data with regards to other datasets such as Craigslist, Zigbang and Ohou. Compute key metrics. (test_metrics_external_dataset.ipynb).
  
