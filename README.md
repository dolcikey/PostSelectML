# PostScriptML 
## A Machine Learning based rating tool for selecting your most ideal raw images so you don't have to. 

### Created by Dolci Key Sanders


## Background 

Having come from a background of both photography and fashion, I have spent a considerable amount of time both behind and infront of the camera. My grandfather was a photography professor and I grew up with a film camera in hand. My grandad was one of the first of his generation to embrace the digital age in photography and started experimenting with digital images when I was in high school. He often looked through my images, critiquing the framing or sometimes prasing the balance of the image or sharp focus. In film photography, image selection happened after film was developed, but in the digital age, after images are downloaded, images go through a process called selection, sometimes even multiple rounds of selection before final images are choses, edited, and exported. This process can take hours and is a pain-staking process, or was. Until now. 

The idea behind this model is that once trained with a vast amount of knowlege to intepret and analyze photography, this Convelutional Neural Network will be able to take a set of raw (.NEF) images before editing and select the images having a probability of idealness of 90% or above. Once selected, you can take those images and directly start editing. Saving hours, time, and money. 

## Business Case

Many businesses rely on content for marketing in the form of photographs. From E-commerce sites, to social media driven businesses, to smaller businesses like wedding photographers, these businesses all rely on a steady stream of photographic content. 

### Usual photography work flow
#### Shoot > Download images > Pre-Selection Process > Selection Process > Edit Images > Export

During the post shoot process, after images are downloaded, many photographers spend hours sorting through hundreds or thousands of images during what is called the selection process. Similar to data science, this can be an iterative process with multiple reviews of the photos for selection. Some businesses and photographers have assistants or interns sort through these images. PostScriptML aims to minimize time spent in the pre-selection and selection processes using a convelutional neural network to filter images and return back the images that meet a certain probability of "idealness" in order to move on to the editing process. 

PostScriptML can minimize time spent on processing by making the selection process an unsupervised task saving time, money, and minimizing the probability of rejecting a great image. 



## Repository Navigation 
Here are all the files found in the PostScriptML Repository.

### README.md - You are Here - General Overview of the Repository
### 01_PostScriptML_Cleaning.ipynb - Image break down and clean up
### 02_PostScriptML_EDA.ipynb - Visualizations and some Exploration of the Images
### 03_PostScriptML_Modeling.ipynb - Model coding notebook
### Scripts - for adding models to the AWS Files, Models
### Visuals - PNG files of all visuals in the EDA and Modeling Notebooks

## Data

I used RAW .NEF (Nikon) images from my archived shoots as well as incorporating 4 new shoots specifically for this project. All images used with the models' consent for this project. The two classes for these images are select (selected) and reject (rejected). 

Data Considerations

1. Class Imbalance

	Due to the nature of selects, there is often a large class imbalance when it comes to selected images verses the rejected images. This set was no different. To deal with class imbalance, all images of the selects sets were augmented by flipping them horizontally. This allowed the image to be different, but the same image to help boost the selected class. 

	Even with augmented images, this set is still very imbalanced, so using weighting methods in Keras also helped mitigate the imbalance issues. 

	<class imbalance>

2. Racial Bias 
	
	To minimize any potential racial bias of the model, I was able to include a mix of Caucasion, Dominican, African American, Asian, and bi-racial (Asian-Hispanic) models. 

	<Racial Breakdown of All Images> <Racial Breakdown of Train Images>
	<Racial Breakdown of Test Images> <Racial Breakdown of Validation Images>

3. Image Size
	
	The nature of .NEF files is that they are 14 bytes to a pixel making for large images. These are the images that professional photographers use to keep all data in an image for editing. Once edited, the image is then compressed via the exporting process to a .jpg or other file type depending on its intended usage. 

	Images in this data set in their RAW form ranged from 9.6 MB to 65.5 MB. All but one set of images came from a Nikon Z7 camera using a XQD memory card, the smaller set from a Nikon D90 using a SD memory card. The smallest set was used in the validation set. 

	Images were stored using an S3 bucket for use in a SageMaker Notebook Instance where the modeling took place. Images were presorted into test, train, and validation folders where each subject(human model) was isolated to only one folder to prevent data leakage. 

	The total data size for all imaages was [insert GB of data for final project here]

4. Personal/Artistic Bias
	
	As I have personally selected what I believe to be most ideal for the data, there could be some artistic bias included in this model. I mainly focused on balance, eye focus, rule of thirds, and symmetry to make decisions on wether or not the photo should qualify as selected. Artistically blurry photos were not included as ideal, and photographers who are more artistic with movement may not benefit from this model. However, most photographers in business cases are looking for similar elements for their clients wether in portrait photography, advertisments, or e-commerce.  

	<selected image> <similar reject image>

	3. Class Imbalance

	Due to the nature of selects, there is often a large class imbalance when it comes to selected images verses the rejected images. This set was no different. To deal with class imbalance, all images of the selects sets were augmented by flipping them horizontally. This allowed the image to be different, but the same image to help boost the selected class. 

	Even with augmented images, this set is still very imbalanced, so using weighting methods in Keras also helped mitigate the imbalance issues. 

	<class imbalance image>

# The Process 

First, I began by gathering data. I did this by gathering .NEF files, creating more images on 4 additonal shoots to supplement my data, and sorting through to label the select and reject classes. Once done, I broke the shoots up into training, test, and validation sets. 


## Cleaning/EDA
	Connecting S3 bucket from AWS to import test, train, and validation files. 
	Augmentation of the selects by horizontally flipping each image. 
	Visualizations of class imbalance, subject models by race.

	Image compression and reshaping. 

## Modeling
	AWS notebook
	Metrics used


# Reproduction Instructions

# Presentation Deck 
	<link here>


# Conclusion



# Future Steps 

## Layers Integration 

I would like to create layers to specifically focus on:

	- Eye Focus
	- Rule of Thirds
	- Symmetry
	- Angles

## Apps and Software Development 

I would like to further expand this model into an app and also eventually work towards building software that can be integrated into a camera for real time analysis of angles, framing, and balance. 

## Sources: 

Photographs by Dolci Key Photography

Image Subjects (Models) without whom, this data would have been hard to find elsewhere:
Kristen Heavy, Samayah Jaramillo, Bethany Chasteen, Joanna Pauline, Skylar Bumgartner, Christina

Image Storage by AWS S3 Buckets
Modeling powered by AWS SageMaker 

Scripts
Code  






