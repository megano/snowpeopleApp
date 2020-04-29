# snowpeopleApp
  
  Do you want to detect a snowman? Great! You've come to the right place. This is the repo for an interactive deep learning image classifier. It suppports detection of 2 categories of snowmen: abominable, and Olaf. To try out the model, visit this URL and follow the instructions you’ll see on the page: https://mybinder.org/v2/gh/megano/snowpeopleApp/master?urlpath=%2Fvoila%2Frender%2Fsnowpeople_voila.ipynb  
  
  Source data: The model is trained on 2 classes, so it works best when applied to differentiating between the abominable snowman and the friendly snowman 'Olaf' from Disney's 'Frozen'. It does work on permutations of classic features, recognizing Olaf when his features appear on a toilet paper roll for example, but won't likely generalize beyond these 2 very specific classes of snowman that it was trained on. 
  
  Why did I make this? Mostly to practice pushing deep learning image classification models into production. Partially to entertain myself & bring a bit of joy to kids and adults trapped indoors during quarentine. You can spend many minutes trying to trick the model, and finding it's edges. Hint: try seeing if it recognizes Olaf when he's making different facial expressions.   
  
  This repo contains code for the model, and a simple front-end hosted on mybinder.org where you can upload your own image and see classification results. Details on the files, and data used for training below. 
 * export.pkl - Reduced model size under 50MB. 
   Built with resnet18, image size reduced to 64px for training, and same 300 images, but lower resolution. Reduced to work within Binder app hosting constraints.
  
 * export-81MB.pkl - Original sized model export here for reference. 
  Built with resnet18, and 300 images from Bing search API: 150 images of abominable snowman, 150 of Olaf. 
  
 * requirements.txt - requirements file lists dependencies. 
  
 * snowpeople_voila.ipynb - Notebook with basic python widgets for UI, and path to model file that Binder uses to generate the production UI. 
  
  
  
  
  
