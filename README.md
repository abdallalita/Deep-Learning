## [Shoe Brand Image Classification](#Problem Statement)
### Problem Statement
The goal of this project is to **classify shoe images** into different three brand categories, **Converse**, **Adidas**, **Nike**, using deep learning techniques (CNN).
Specifically, we want to train a model that can accurately predict the brand category of an input image from a pre-defined set of class labels.

---

### Dataset
The dataset used for this project is a collection of labeled images. The dataset was obtained by downloading images from Google images.
The images with a .webp format were transformed into .jpg images. The obtained images had a resolution of 240x240 pixels in RGB color model(3 colour channels).
The test-train-split ratio is 0.14, with the test dataset containing 114 images and the train dataset containing 711. The dataset
is in standard image classification format, containing
separate classes of images in seperate directories titled with a particular class name, all images of Nike are contained in the **Nike/** directory and so on.

---

### Summarization
I performed Transformation or **Data Augmentation** , using `torchvision.transforms` inbuilt functions. It is the process of altering your data in such a way 
that you **artificially increase the diversity**
of your training set. Training a model on this artificially altered dataset hopefully results in a model that is capable of **better generalization**
(the patterns it learns are more robust to future unseen examples).
I loaded the image data into **Datasets** format capable of being used with PyTorch, using the `torchvision.datasets.ImageFolder` function and later turn 
the loaded images datasets into **DataLoader's**, which makes them iterable (batchify) so a model can go through learn the relationships between samples and targets 
(features and labels), using ` torch.utils.data.DataLoader` function.
I created a TinyVGG model class from the [https://poloclub.github.io/cnn-explainer/](CNN Explainer), training and test loop functions to train our model on
the training data and evaluate
our model on the testing data, using 10 **epochs**.
As for an **optimizer** and **loss function**, I used `torch.nn.CrossEntropyLoss()` since I'm working with **multi-class classification** data
and `torch.optim.Adam()` with a **learning rate** of 1e-3 respecitvely.


---

### Results
Our model perfomed pretty poorly with both train and test accuracy of around 30% - 35% and both the losses decreased very slightly. Ideally we would like the loss to decarese
to zero and the accuracy to increase to 100%, however the accuracy is way below half value or 50%.
By plotting the **Loss Curves**, I found out that our model was kind of sporadic (the metrics go up and down sharply).
The train and tes loss metric were much lower, however for the accuracy curve metric, the test accuracy was a bit higher than that for the train accuracy.

