# TransMotion

 [ [Quick Overview](https://gaxler.github.io/transmotion/book/demo.html) | [Docs](https://gaxler.github.io/transmotion/doc/) | [Original TPS Repo](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model)] 
 

The code that came with the
[FOMM](https://github.com/AliaksandrSiarohin/first-order-model) paper
became a bit of a standard in this area of research, but the code lacks
annotation as to what kind of objects it passes around.

This lack of annotation and documentation makes it harder to understand
the code and scarier to tweak it.

This project is here to document what’s going on there. Key focus is on
type annotaoitn and use of `dataclasses` to be more explicit as to what
is being passed around and `einops` to be clearer about the shapes of
tensors.

<a href="https://colab.research.google.com/github/gaxler/transmotion/blob/master/colab_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Eventually you should get something like this:

![](static/out.gif)
 

#### This is still a work in progress. Most notably you’ll find missing

-   data loading functionality
-   training loop documentation


