# Background 
Recently while taking Practical Deep Learning for Coders I hatched a plan to get my nieces curious about machine learning. They’re obsessed with the movie “Frozen” and love puzzles, so I created an image classifier that could identify the friendly snowman Olaf, and the abominable snowman. I hosted it on the web so they could upload their own images and try to “trick” the deep learning classifier into labeling an image of Olaf as the abominable snowman.  

# Goal
Using the fast.ai library, select & fine tune a convolutional neural network (CNN) architecture to identify 2 classes of images. Host the model on the web so others can try it out & upload their own images for it to classify. Use a simple hosting service like Binder (binder.org) to minimize time spent setting up hosting.  

# Approach
I selected resnet18 because it’s one of the smallest CNN architectures. I sourced training data from 300 images, 150 for each class, from the Bing search API. The model detects Olaf with x% precision and y% recall and detects the abominable snowman with x% precision and y% recall on the top 150 images of each class on Bing. 

# Outcome
The project was a success. The classifier is live online at https://mybinder.org/v2/gh/megano/snowpeopleApp/master?urlpath=%2Fvoila%2Frender%2Fsnowpeople_voila.ipynb. To try out the model, visit the URL and follow the instructions you’ll see on the page. 
  
# Limitations & Repo Navitation 
The model is trained on 2 classes. It does work on permutations of features, recognizing Olaf when his features appear on a toilet paper roll for example, but won't likely generalize beyond these 2 very specific classes of snowman that it was trained on. This repo contains code for the model, and a simple front-end hosted on mybinder.org. Details on the files, and data used for training below. 
 * export.pkl - Reduced model size under 50MB. 
   Built with resnet18, image size reduced to 64px for training, and same 300 images, but lower resolution. Reduced to work within Binder app hosting constraints.
  
 * export-81MB.pkl - Original sized model export here for reference. 
  Built with resnet18, and 300 images from Bing search API: 150 images of abominable snowman, 150 of Olaf. 
  
 * requirements.txt - Lists dependencies. 
  
 * snowpeople_voila.ipynb - Notebook with basic python widgets for UI, and path to the model file that Binder uses to generate the production UI. 
