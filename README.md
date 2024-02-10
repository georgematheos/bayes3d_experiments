# Bayes3D Experiments

I'm using this repo for some experiments with Bayes3D.

For visitors, I recommend looking at:
- [experiments/01_tutorial-model.ipynb](experiments/01_tutorial-model.ipynb)
    - This is a walk-through of a variant of the Bayes3D scene graph model.
    - For GenJAX enthusiasts: this also serves as a tutorial on how to use the Masking interface
    in GenJAX to implement a model with variable structure and size (here, a variable
    number of objects).
- [experiments/02_tutorial-inference.ipynb](experiments/02_tutorial-inference.ipynb)
    - This walks through a MLE inference procedure for the model in the previous notebook,
    and demonstrates it by fitting two synthetic scenes, one of several YCB objects
    on a table (this scene comes from a pybullet environment), and one consisting
    of several cubes stacked atop each other.